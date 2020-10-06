import argparse
import torch
from utils import *
import os
from aveNet import aveNet
from mir_eval.separation import bss_eval_sources
from pystoi import stoi
from pesq import pesq

MAX_INT16 = np.iinfo(np.int16).max

def write_wav(fname, samps, sampling_rate=16000, normalize=True):
    """
    Write wav files in int16, support single/multi-channel
    """
    # for multi-channel, accept ndarray [Nsamples, Nchannels]
    samps = np.divide(samps, np.max(np.abs(samps)))

    # same as MATLAB and kaldi
    if normalize:
        samps = samps * MAX_INT16
        samps = samps.astype(np.int16)
    fdir = os.path.dirname(fname)
    if fdir and not os.path.exists(fdir):
        os.makedirs(fdir)
    # NOTE: librosa 0.6.0 seems could not write non-float narray
    #       so use scipy.io.wavfile instead
    wavfile.write(fname, sampling_rate, samps)

def SDR(est, egs, mix):
    '''
        calculate SDR
        est: Network generated audio
        egs: Ground Truth
    '''
    sdr, _, _, _ = bss_eval_sources(egs, est)
    mix_sdr, _, _, _ = bss_eval_sources(egs, mix)
    return float(sdr-mix_sdr)

class dataset(data.Dataset):
    def __init__(self,
                mix_lst_path,
                audio_direc,
                visual_direc,
                mixture_direc,
                batch_size=1,
                partition='test',
                sampling_rate=16000,
                mix_no=2):

        self.minibatch =[]
        self.audio_direc = audio_direc
        self.visual_direc = visual_direc
        self.mixture_direc = mixture_direc
        self.sampling_rate = sampling_rate
        self.partition = partition
        self.C=mix_no

        mix_csv=open(mix_lst_path).read().splitlines()
        self.mix_lst=list(filter(lambda x: x.split(',')[0]==partition, mix_csv))

    def __getitem__(self, index):
        line = self.mix_lst[index]

        mixture_path=self.mixture_direc+self.partition+'/'+ line.replace(',','_').replace('/', '_') +'.wav'
        _, mixture = wavfile.read(mixture_path)
        mixture = self._audio_norm(mixture)
        
        min_length = mixture.shape[0]

        line=line.split(',')
        c=0
        audio_path=self.audio_direc+line[c*4+1]+'/'+line[c*4+2]+'/'+line[c*4+3]+'.wav'
        _, audio = wavfile.read(audio_path)
        audio = self._audio_norm(audio[:min_length])

        # read visual
        visual_path=self.visual_direc+line[c*4+1]+'/'+line[c*4+2]+'/'+line[c*4+3]+'.npy'
        visual = np.load(visual_path)
        length = math.floor(min_length/self.sampling_rate*25)
        visual = visual[:length,...]
        a = visual.shape[0]
        if visual.shape[0] < length:
            visual = np.pad(visual, ((0,int(length - visual.shape[0])),(0,0)), mode = 'edge')

        embedding = 0

        return mixture, audio, visual, embedding, (line[c*4+2]+'/'+line[c*4+3])

    def __len__(self):
        return len(self.mix_lst)

    def _audio_norm(self,audio):
        return np.divide(audio, np.max(np.abs(audio)))

def main(args):
    # Model
    model = aveNet(args.N, args.L, args.B, args.H, args.P, args.X, args.R,
                        args.C, 800)

    model = model.cuda()
    pretrained_model = torch.load('%smodel_dict.pt' % args.continue_from, map_location='cpu')['model']

    state = model.state_dict()
    for key in state.keys():
        pretrain_key = 'module.' + key
        if pretrain_key in pretrained_model.keys():
            state[key] = pretrained_model[pretrain_key]
    model.load_state_dict(state)

    datasets = dataset(
                mix_lst_path=args.mix_lst_path,
                audio_direc=args.audio_direc,
                visual_direc=args.visual_direc,
                mixture_direc=args.mixture_direc,
                mix_no=args.C)

    test_generator = data.DataLoader(datasets,
            batch_size = 1,
            shuffle = False,
            num_workers = args.num_workers)

    
    model.eval()
    with torch.no_grad():
        avg_sisnri = 0
        avg_sdri = 0
        avg_pesqi = 0
        avg_stoii = 0
        for i, (a_mix, a_tgt, v_tgt, a_emb, fname) in enumerate(tqdm.tqdm(test_generator)):
            a_mix = a_mix.cuda().squeeze().float().unsqueeze(0)
            a_tgt = a_tgt.cuda().squeeze().float().unsqueeze(0)
            v_tgt = v_tgt.cuda().squeeze().float().unsqueeze(0)
            a_emb = a_emb.cuda().squeeze()

            est_speaker, estimate_source = model(a_mix, v_tgt)

            sisnr_mix = cal_SISNR(a_tgt, a_mix)
            sisnr_est = cal_SISNR(a_tgt, estimate_source)
            sisnri = sisnr_est - sisnr_mix
            avg_sisnri += sisnri
            print(sisnri)

            estimate_source = estimate_source.squeeze().cpu().numpy()
            a_tgt = a_tgt.squeeze().cpu().numpy()
            a_mix = a_mix.squeeze().cpu().numpy()


            # save_path = "/home/panzexu/samples/avaNet1/" + fname[0] +'.wav'
            # write_wav(save_path, estimate_source)

            save_path = "/home/panzexu/samples/avaNet1_emb/test/" + fname[0] +'.npy'
            # print(save_path)
            if not os.path.exists(save_path.rsplit('/', 1)[0]):
                os.makedirs(save_path.rsplit('/', 1)[0])
            np.save(save_path, est_speaker.cpu().numpy())

            # avg_sdri += SDR(estimate_source, a_tgt, a_mix)
            # # print(SDR(estimate_source, a_tgt, a_mix))
            # avg_pesqi += (pesq(16000, a_tgt, estimate_source, 'wb') - pesq(16000, a_tgt, a_mix, 'wb'))
            # # print(pesq(16000, a_tgt, estimate_source, 'wb') - pesq(16000, a_tgt, a_mix, 'wb'))
            # avg_stoii += (stoi(a_tgt, estimate_source, 16000, extended=False) - stoi(a_tgt, a_mix, 16000, extended=False))
            # # print((stoi(a_tgt, estimate_source, 16000, extended=False) - stoi(a_tgt, a_mix, 16000, extended=False)))
            # # if sisnri < 10 and sisnr_est < 10:
            # #     wavfile.write('/home/panzexu/listen/ava1/%s_tgt.wav' %i, 16000, a_tgt)
            # #     wavfile.write('/home/panzexu/listen/ava1/%s_mix_%s.wav' %(i, sisnr_mix.item()), 16000, a_mix)
            # #     wavfile.write('/home/panzexu/listen/ava1/%s_est_%s.wav' %(i, sisnr_est.item()), 16000, np.divide(estimate_source, np.max(np.abs(estimate_source))))
            # # # if i >=10:
            # #     break
        
        avg_sisnri = avg_sisnri / (i+1)
        avg_sdri = avg_sdri / (i+1)
        avg_pesqi = avg_pesqi / (i+1)
        avg_stoii = avg_stoii / (i+1)
        print(avg_sisnri)
        print(avg_sdri)
        print(avg_pesqi)
        print(avg_stoii)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("avConv-tasnet")
    
    # Dataloader
    # parser.add_argument('--mix_lst_path', type=str, default='/home/panzexu/datasets/voxceleb2/audio_mixture/2_mix_min_800/mixture_data_list_2mix.csv',
    #                     help='directory including train data')
    # parser.add_argument('--audio_direc', type=str, default='/home/panzexu/datasets/voxceleb2/audio_clean/',
    #                     help='directory including validation data')
    # parser.add_argument('--visual_direc', type=str, default='/home/panzexu/datasets/voxceleb2/visual_embedding/lip/',
    #                     help='directory including test data')
    # parser.add_argument('--mixture_direc', type=str, default='/home/panzexu/datasets/voxceleb2/audio_mixture/2_mix_min_800/',
    #                     help='directory of audio')
    # parser.add_argument('--mix_lst_path', type=str, default='/home/panzexu/datasets/LRS3/audio_mixture/2_mix_min/mixture_data_list_2mix.csv',
    #                     help='directory including train data')
    # parser.add_argument('--audio_direc', type=str, default='/home/panzexu/datasets/LRS3/audio_clean/',
    #                     help='directory including validation data')
    # parser.add_argument('--visual_direc', type=str, default='/home/panzexu/datasets/LRS3/visual_embedding/lip/',
    #                     help='directory including test data')
    # parser.add_argument('--mixture_direc', type=str, default='/home/panzexu/datasets/LRS3/audio_mixture/2_mix_min/',
    #                     help='directory of audio')
    # # # grid
    # TCDTIMITDataset
    # LRS2
    # LRS3
    # avspeech_subset

    # Log and Visulization
    # parser.add_argument('--continue_from', type=str, 
    #     default='/home/panzexu/workspace/avss_speaker_embedding/log/voxceleb_800/2_mix/avaNet_2020-09-29(19:27:48)_wo/')
    # parser.add_argument('--continue_from', type=str, 
    #     default='/home/panzexu/workspace/avss_speaker_embedding/src/avaNet1/logs/avaNet_2020-09-29(19:27:48)/')

    # parser.add_argument('--mix_lst_path', type=str, default='/home/panzexu/datasets/voxceleb2/audio_mixture/3_mix_min_800/mixture_data_list_3mix.csv',
    #                     help='directory including train data')
    # parser.add_argument('--audio_direc', type=str, default='/home/panzexu/datasets/voxceleb2/audio_clean/',
    #                     help='directory including validation data')
    # parser.add_argument('--visual_direc', type=str, default='/home/panzexu/datasets/voxceleb2/visual_embedding/lip/',
    #                     help='directory including test data')
    # parser.add_argument('--mixture_direc', type=str, default='/home/panzexu/datasets/voxceleb2/audio_mixture/3_mix_min_800/',
    #                     help='directory of audio')
    parser.add_argument('--mix_lst_path', type=str, default='/home/panzexu/datasets/LRS3/audio_mixture/3_mix_min/mixture_data_list_3mix.csv',
                        help='directory including train data')
    parser.add_argument('--audio_direc', type=str, default='/home/panzexu/datasets/LRS3/audio_clean/',
                        help='directory including validation data')
    parser.add_argument('--visual_direc', type=str, default='/home/panzexu/datasets/LRS3/visual_embedding/lip/',
                        help='directory including test data')
    parser.add_argument('--mixture_direc', type=str, default='/home/panzexu/datasets/LRS3/audio_mixture/3_mix_min/',
                        help='directory of audio')

    parser.add_argument('--continue_from', type=str, 
        default='/home/panzexu/workspace/avss_speaker_embedding/log/voxceleb_800/3_mix/avaNet_2020-10-04(16:49:32)/')
    

    # Training    
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to generate minibatch')

    # Model hyperparameters
    parser.add_argument('--L', default=40, type=int,
                        help='Length of the filters in samples (80=5ms at 16kHZ)')
    parser.add_argument('--N', default=256, type=int,
                        help='Number of filters in autoencoder')
    parser.add_argument('--B', default=256, type=int,
                        help='Number of channels in bottleneck 1 × 1-conv block')
    parser.add_argument('--C', type=int, default=2,
                        help='number of speakers to mix')
    parser.add_argument('--H', default=512, type=int,
                        help='Number of channels in convolutional blocks')
    parser.add_argument('--P', default=3, type=int,
                        help='Kernel size in convolutional blocks')
    parser.add_argument('--X', default=8, type=int,
                        help='Number of convolutional blocks in each repeat')
    parser.add_argument('--R', default=4, type=int,
                        help='Number of repeats')

    args = parser.parse_args()

    main(args)