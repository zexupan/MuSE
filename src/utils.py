import numpy as np
import math
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.utils.data as data
import scipy.io.wavfile as wavfile
from itertools import permutations
from apex import amp
import tqdm
import os

EPS = 1e-6

class dataset(data.Dataset):
    def __init__(self,
                speaker_dict,
                mix_lst_path,
                audio_direc,
                visual_direc,
                mixture_direc,
                batch_size,
                partition='test',
                audio_only=False,
                sampling_rate=16000,
                max_length=4,
                mix_no=2):

        self.minibatch =[]
        self.audio_only = audio_only
        self.audio_direc = audio_direc
        self.visual_direc = visual_direc
        self.mixture_direc = mixture_direc
        self.sampling_rate = sampling_rate
        self.partition = partition
        self.max_length = max_length
        self.C=mix_no
        self.speaker_id=speaker_dict

        mix_lst=open(mix_lst_path).read().splitlines()
        mix_lst=list(filter(lambda x: x.split(',')[0]==partition, mix_lst))
        
        assert (batch_size%self.C) == 0, "input batch_size should be multiples of mixture speakers"

        self.batch_size = int(batch_size/self.C )
        sorted_mix_lst = sorted(mix_lst, key=lambda data: float(data.split(',')[-1]), reverse=True)
        start = 0
        while True:
            end = min(len(sorted_mix_lst), start + self.batch_size)
            self.minibatch.append(sorted_mix_lst[start:end])
            if end == len(sorted_mix_lst):
                break
            start = end

    def __getitem__(self, index):
        batch_lst = self.minibatch[index]
        min_length = int(float(batch_lst[-1].split(',')[-1])*self.sampling_rate)

        mixtures=[]
        audios=[]
        visuals=[]
        speakers=[]
        for line in batch_lst:
            mixture_path=self.mixture_direc+self.partition+'/'+ line.replace(',','_').replace('/','_')+'.wav'
            _, mixture = wavfile.read(mixture_path)
            mixture = self._audio_norm(mixture[:min_length])
            
            line=line.split(',')
            for c in range(self.C):
                # read target audio
                audio_path=self.audio_direc+line[c*4+1]+'/'+line[c*4+2]+'/'+line[c*4+3]+'.wav'
                _, audio = wavfile.read(audio_path)
                audios.append(self._audio_norm(audio[:min_length]))

                # read target audio id
                if self.partition == 'test':
                    speakers.append(0)
                else: speakers.append(self.speaker_id[line[c*4+2]])

                # read target visual reference
                visual_path=self.visual_direc+line[c*4+1]+'/'+line[c*4+2]+'/'+line[c*4+3]+'.npy'
                visual = np.load(visual_path)
                length = math.floor(min_length/self.sampling_rate*25)
                visual = visual[:length,...]
                a = visual.shape[0]
                if visual.shape[0] < length:
                    visual = np.pad(visual, ((0,int(length - visual.shape[0])),(0,0)), mode = 'edge')
                visuals.append(visual)

                # read overlapped speech
                mixtures.append(mixture)
        
        return np.asarray(mixtures)[...,:self.max_length*self.sampling_rate], \
                np.asarray(audios)[...,:self.max_length*self.sampling_rate], \
                np.asarray(visuals)[...,:self.max_length*25,:], \
                np.asarray(speakers)

    def __len__(self):
        return len(self.minibatch)

    def _audio_norm(self,audio):
        return np.divide(audio, np.max(np.abs(audio)))


class DistributedSampler(data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # indices = torch.randperm(len(self.dataset), generator=g).tolist()
            ind = torch.randperm(int(len(self.dataset)/self.num_replicas), generator=g)*self.num_replicas
            indices = []
            for i in range(self.num_replicas):
                indices = indices + (ind+i).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        # indices = indices[self.rank:self.total_size:self.num_replicas]
        indices = indices[self.rank*self.num_samples:(self.rank+1)*self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

def get_dataloader(args, partition):
    datasets = dataset(
                speaker_dict =args.speaker_dict,
                mix_lst_path=args.mix_lst_path,
                audio_direc=args.audio_direc,
                visual_direc=args.visual_direc,
                mixture_direc=args.mixture_direc,
                batch_size=args.batch_size,
                max_length=args.max_length,
                partition=partition,
                mix_no=args.C)

    sampler = DistributedSampler(
        datasets,
        num_replicas=args.world_size,
        rank=args.local_rank) if args.distributed else None

    generator = data.DataLoader(datasets,
            batch_size = 1,
            shuffle = (sampler is None),
            num_workers = args.num_workers,
            sampler=sampler)

    return sampler, generator

@amp.float_function
def cal_SISNR(source, estimate_source):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        source: torch tensor, [batch size, sequence length]
        estimate_source: torch tensor, [batch size, sequence length]
    Returns:
        SISNR, [batch size]
    """
    assert source.size() == estimate_source.size()

    # Step 1. Zero-mean norm
    source = source - torch.mean(source, axis = -1, keepdim=True)
    estimate_source = estimate_source - torch.mean(estimate_source, axis = -1, keepdim=True)

    # Step 2. SI-SNR
    # s_target = <s', s>s / ||s||^2
    ref_energy = torch.sum(source ** 2, axis = -1, keepdim=True) + EPS
    proj = torch.sum(source * estimate_source, axis = -1, keepdim=True) * source / ref_energy
    # e_noise = s' - s_target
    noise = estimate_source - proj
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    ratio = torch.sum(proj ** 2, axis = -1) / (torch.sum(noise ** 2, axis = -1) + EPS)
    sisnr = 10 * torch.log10(ratio + EPS)

    return sisnr


if __name__ == '__main__':
    datasets = dataset(
                mix_lst_path='/home/panzexu/datasets/LRS3/audio/2_mix_min/mixture_data_list_2mix.csv',
                audio_direc='/home/panzexu/datasets/LRS3/audio/Audio/',
                visual_direc='/home/panzexu/datasets/LRS3/lip/',
                mixture_direc='/home/panzexu/datasets/LRS3/audio/2_mix_min/',
                batch_size=8,
                partition='train')
    data_loader = data.DataLoader(datasets,
                batch_size = 1,
                shuffle= True,
                num_workers = 4)

    for a_mix, a_tgt, v_tgt, speakers in tqdm.tqdm(data_loader):
        # print(a_mix.squeeze().size())
        # print(a_tgt.squeeze().size())
        # print(v_tgt.squeeze().size())
        pass

    # a = np.ones((24,512))
    # print(a.shape)
    # a = np.pad(a, ((0,-1), (0,0)), 'edge')
    # print(a.shape)

    # a = np.random.rand(2,2,3)
    # print(a)
    # a = a.reshape(4,3)
    # print(a)