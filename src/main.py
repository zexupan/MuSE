import argparse
import torch
from utils import *
import os
from model import muse
from solver import Solver


def main(args):
    if args.distributed:
        torch.manual_seed(0)
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # speaker id assignment
    mix_lst=open(args.mix_lst_path).read().splitlines()
    train_lst=list(filter(lambda x: x.split(',')[0]=='train', mix_lst))
    IDs = 0
    speaker_dict={}
    for line in train_lst:
        for i in range(2):
            ID = line.split(',')[i*4+2]
            if ID not in speaker_dict:
                speaker_dict[ID]=IDs
                IDs += 1
    args.speaker_dict=speaker_dict
    args.speakers=len(speaker_dict)

    # Model
    model = muse(args.N, args.L, args.B, args.H, args.P, args.X, args.R,
                        args.C, args.speakers)

    if (args.distributed and args.local_rank ==0) or args.distributed == False:
        print("started on " + args.log_name + '\n')
        print(args)
        print("\nTotal number of parameters: {} \n".format(sum(p.numel() for p in model.parameters())))
        print(model)

    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_sampler, train_generator = get_dataloader(args,'train')
    _, val_generator = get_dataloader(args, 'val')
    _, test_generator = get_dataloader(args, 'test')
    args.train_sampler=train_sampler

    solver = Solver(args=args,
                model = model,
                optimizer = optimizer,
                train_data = train_generator,
                validation_data = val_generator,
                test_data = test_generator) 
    solver.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("avConv-tasnet")
    
    # Dataloader
    parser.add_argument('--mix_lst_path', type=str, default='/home/panzexu/datasets/LRS2/audio/2_mix_min/mixture_data_list_2mix.csv',
                        help='directory including train data')
    parser.add_argument('--audio_direc', type=str, default='/home/panzexu/datasets/LRS2/audio/Audio/',
                        help='directory including validation data')
    parser.add_argument('--visual_direc', type=str, default='/home/panzexu/datasets/LRS2/lip/',
                        help='directory including test data')
    parser.add_argument('--mixture_direc', type=str, default='/home/panzexu/datasets/LRS2/audio/2_mix_min/',
                        help='directory of audio')
    parser.add_argument('--target_Ae_direc', type=str, default='/home/panzexu/datasets/LRS2/xvector_min/',
                        help='directory of audio embedding')


    # Training    
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size')
    parser.add_argument('--max_length', default=6, type=int,
                        help='max_length of mixture in training')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to generate minibatch')
    parser.add_argument('--epochs', default=100, type=int,
                        help='Number of maximum epochs')

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

    # optimizer
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Init learning rate')
    parser.add_argument('--max_norm', default=5, type=float,
                    help='Gradient norm threshold to clip')


    # Log and Visulization
    parser.add_argument('--log_name', type=str, default=None,
                        help='the name of the log')
    parser.add_argument('--use_tensorboard', type=int, default=0,
                        help='Whether to use use_tensorboard')
    parser.add_argument('--continue_from', type=str, default='',
                        help='Whether to use use_tensorboard')

    # Distributed training
    parser.add_argument('--opt-level', default='O0', type=str)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--patch_torch_functions', type=str, default=None)

    args = parser.parse_args()

    args.distributed = False
    args.world_size = 1
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        args.world_size = int(os.environ['WORLD_SIZE'])

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."
    
    main(args)
