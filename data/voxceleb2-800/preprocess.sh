#!/bin/bash 

direc=/home/panzexu/datasets/voxceleb2/

data_direc=${direc}orig/

train_samples=20000 # no. of train mixture samples simulated
val_samples=5000 # no. of validation mixture samples simulated
test_samples=3000 # no. of test mixture samples simulated
C=2 # no. of speakers in the mixture
mix_db=10 # random db ratio from -10 to 10db
mixture_data_list=mixture_data_list_${C}mix.csv #mixture datalist
sampling_rate=16000 # audio sampling rate
min_length=4 # minimum length of audio

audio_data_direc=${direc}audio_clean/ # Target audio saved directory
mixture_audio_direc=${direc}audio_mixture/${C}_mix_min_800/ # Audio mixture saved directory
visual_frame_direc=${direc}face/ # The visual saved directory
lip_embedding_direc=${direc}lip/ # The lip embedding saved directory

# stage 1: Remove repeated datas in pretrain and train set, extract audio from mp4, create mixture list
echo 'stage 1: create mixture list'
python 1_create_mixture_list.py \
--data_direc $data_direc \
--C $C \
--mix_db $mix_db \
--train_samples $train_samples \
--val_samples $val_samples \
--test_samples $test_samples \
--audio_data_direc $audio_data_direc \
--min_length $min_length \
--sampling_rate $sampling_rate \
--mixture_data_list $mixture_data_list \

# stage 2: create audio mixture from list
echo 'stage 2: create mixture audios'
python 2_create_mixture.py \
--C $C \
--audio_data_direc $audio_data_direc \
--mixture_audio_direc $mixture_audio_direc \
--mixture_data_list $mixture_data_list \

# stage 3: create lip embedding
echo 'stage 3: create lip embedding' 
python 3_create_lip_embedding.py \
--video_data_direc $data_direc \
--visual_frame_direc $visual_frame_direc \
--lip_embedding_direc $lip_embedding_direc
