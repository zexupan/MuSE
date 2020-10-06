import os
import numpy as np 
import argparse
import tqdm
import scipy.io.wavfile as wavfile

MAX_INT16 = np.iinfo(np.int16).max

def write_wav(fname, samps, sampling_rate=16000, normalize=True):
	"""
	Write wav files in int16, support single/multi-channel
	"""
	# for multi-channel, accept ndarray [Nsamples, Nchannels]
	if samps.ndim != 1 and samps.shape[0] < samps.shape[1]:
		samps = np.transpose(samps)
		samps = np.squeeze(samps)
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


def read_wav(fname, normalize=True):
    """
    Read wave files using scipy.io.wavfile(support multi-channel)
    """
    # samps_int16: N x C or N
    #   N: number of samples
    #   C: number of channels
    sampling_rate, samps_int16 = wavfile.read(fname)
    # N x C => C x N
    samps = samps_int16.astype(np.float)
    # tranpose because I used to put channel axis first
    if samps.ndim != 1:
        samps = np.transpose(samps)
    # normalize like MATLAB and librosa
    if normalize:
        samps = samps / MAX_INT16
    return sampling_rate, samps

def main(args):
	# create mixture
	mixture_data_list = open(args.mixture_data_list).read().splitlines()
	print(len(mixture_data_list))
	for line in tqdm.tqdm(mixture_data_list,desc = "Generating audio mixtures"):
		data = line.split(',')
		save_direc=args.mixture_audio_direc+data[0]+'/'
		if not os.path.exists(save_direc):
			os.makedirs(save_direc)
		
		mixture_save_path=save_direc+line.replace(',','_').replace('/','_') +'.wav'
		if os.path.exists(mixture_save_path):
			continue

		# read target audio
		_, audio_mix=read_wav(args.audio_data_direc+data[1]+'/'+data[2]+'/'+data[3]+'.wav')
		target_power = np.linalg.norm(audio_mix, 2)**2 / audio_mix.size

		# read inteference audio
		for c in range(1, args.C):
			audio_path=args.audio_data_direc+data[c*4+1]+'/'+data[c*4+2]+'/'+data[c*4+3]+'.wav'
			_, audio = read_wav(audio_path)
			intef_power = np.linalg.norm(audio, 2)**2 / audio.size

			# audio = audio_norm(audio)
			scalar = (10**(float(data[c*4+4])/20)) * np.sqrt(target_power/intef_power)
			audio = audio * scalar

			# truncate long audio with short audio in the mixture
			if audio_mix.shape[0] > audio.shape[0]:
				audio_mix = audio_mix[:audio.shape[0]] + audio
			else: audio_mix = audio_mix + audio[:audio_mix.shape[0]]

		audio_mix = np.divide(audio_mix, np.max(np.abs(audio_mix)))
		write_wav(mixture_save_path, audio_mix)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='LRS2 dataset')
	parser.add_argument('--C', type=int)
	parser.add_argument('--audio_data_direc', type=str)
	parser.add_argument('--mixture_audio_direc', type=str)
	parser.add_argument('--mixture_data_list', type=str)
	args = parser.parse_args()
	main(args)