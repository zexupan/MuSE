import os
import numpy as np 
import argparse
import csv
import tqdm
import librosa
import scipy.io.wavfile as wavfile
from multiprocessing import Pool

np.random.seed(0)

def extract_wav_from_mp4(line):
	# Extract .wav file from mp4
	video_from_path=args.data_direc + line[0]+'/'+line[1]+'/'+line[2]+'.mp4'
	audio_save_path=args.audio_data_direc + line[0]+'/'+line[1]+'/'+line[2]+'.wav'
	if not os.path.exists(audio_save_path.rsplit('/', 1)[0]):
		os.makedirs(audio_save_path.rsplit('/', 1)[0])

	if not os.path.exists(audio_save_path):
		os.system("ffmpeg -i %s %s"%(video_from_path, audio_save_path))
	sr, audio = wavfile.read(audio_save_path)
	assert sr==args.sampling_rate , "sampling_rate mismatch"
	sample_length = audio.shape[0]
	return sample_length # In seconds


def main(args):
	# read the datalist and separate into train, val and test set
	train_list=[]
	val_list=[]
	test_list=[]
	tmp_list=[]

	print("Gathering file names")

	# Get test set list of audios
	for path, dirs ,files in os.walk(args.data_direc + 'test/'):
		for filename in files:
			if filename[-4:] =='.mp4':
				ln = [path.split('/')[-3], path.split('/')[-2], path.split('/')[-1] +'/'+ filename.split('.')[0]]
				sample_length = extract_wav_from_mp4(ln)
				if sample_length < args.min_length*args.sampling_rate: continue
				ln += [sample_length/args.sampling_rate]
				test_list.append(ln)

	# Get train set list of audios
	for path, dirs ,files in os.walk(args.data_direc + 'train/'):
		for filename in files:
			if filename[-4:] =='.mp4':
				ln = [path.split('/')[-3], path.split('/')[-2], path.split('/')[-1] +'/'+ filename.split('.')[0]]
				sample_length = extract_wav_from_mp4(ln)
				if sample_length < args.min_length*args.sampling_rate: continue
				ln += [sample_length/args.sampling_rate]
				tmp_list.append(ln)
	# print(len(tmp_list))

	# Sort the speakers with the number of utterances in pretrain set
	speakers = {}
	for ln in tmp_list:
		ID = ln[1]
		if ID not in speakers:
			speakers[ID] = 1
		else: speakers[ID] +=1
	sort_speakers = sorted(speakers.items(), key=lambda x: x[1], reverse=True)

	# Get the 1600 speakers with the most no. of utterances
	train_speakers={}
	for i, (ID) in enumerate(sort_speakers):
		if i == 800:
			break
		train_speakers[ID[0]] = 0

	for ln in tmp_list:
		ID = ln[1]
		if ID in train_speakers:
			if train_speakers[ID] < 12:
				val_list.append(ln)
				train_speakers[ID] +=1
			elif train_speakers[ID] >=62:
				continue
			else:
				train_list.append(ln)
				train_speakers[ID] +=1


	# Create mixture list
	print("Creating mixture list")
	f=open(args.mixture_data_list,'w')
	w=csv.writer(f)


	# create test set and validation set
	for data_list in [train_list, val_list]:
		if len(data_list)<20000:
			data='val'
			length = args.val_samples
		else:
			data='train'
			length = args.train_samples

		count_list=[]
		for ln in data_list:
			if not ln[1] in count_list:
				count_list.append(ln[1])
		print("In %s list: %s speakers, %s utterances"%(data, len(count_list), len(data_list)))

		cache_list = data_list[:]
		count = 0
		while (len(data_list) >= args.C):
			mixtures=[data]
			shortest = 200
			cache = []
			while len(cache) < args.C:
				idx = np.random.randint(0, len(data_list))
				if data_list[idx][1] in cache:
					continue
				cache.append(data_list[idx][1])
				mixtures = mixtures + list(data_list[idx])
				if float(mixtures[-1]) < shortest: shortest = float(mixtures[-1])
				del mixtures[-1]
				if len(cache)==1: db_ratio =0
				else: db_ratio = np.random.uniform(-args.mix_db,args.mix_db)
				mixtures.append(db_ratio)
				data_list.pop(idx)
			mixtures.append(shortest)
			w.writerow(mixtures)
			count +=1
			if count == length:
				break

		if count < length:
			for j in range(length-count):
				mixtures=[data]
				shortest = 200
				cache = []
				while len(cache) < args.C:
					idx = np.random.randint(0, len(cache_list))
					if cache_list[idx][1] in cache:
						continue
					cache.append(cache_list[idx][1])
					mixtures = mixtures + list(cache_list[idx])
					if float(mixtures[-1]) < shortest: shortest = float(mixtures[-1])
					del mixtures[-1]
					if len(cache)==1: db_ratio =0
					else: db_ratio = np.random.uniform(-args.mix_db,args.mix_db)
					mixtures.append(db_ratio)
				mixtures.append(shortest)
				w.writerow(mixtures)


	data_list=test_list
	data='test'
	length = args.test_samples
	count_list=[]
	for ln in data_list:
		if not ln[1] in count_list:
			count_list.append(ln[1])
	print("In %s list: %s speakers, %s utterances"%(data, len(count_list), len(data_list)))

	for _ in range(length):
		mixtures=[data]
		shortest = 200
		cache = []
		while len(cache) < args.C:
			idx = np.random.randint(0, len(data_list))
			if data_list[idx][1] in cache:
				continue
			cache.append(data_list[idx][1])
			mixtures = mixtures + list(data_list[idx])
			if float(mixtures[-1]) < shortest: shortest = float(mixtures[-1])
			del mixtures[-1]
			if len(cache)==1: db_ratio =0
			else: db_ratio = np.random.uniform(-args.mix_db,args.mix_db)
			mixtures.append(db_ratio)
		mixtures.append(shortest)
		w.writerow(mixtures)


	f.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='LRS2 dataset')
	parser.add_argument('--data_direc', type=str)
	parser.add_argument('--C', type=int)
	parser.add_argument('--mix_db', type=float)
	parser.add_argument('--train_samples', type=int)
	parser.add_argument('--val_samples', type=int)
	parser.add_argument('--test_samples', type=int)
	parser.add_argument('--audio_data_direc', type=str)
	parser.add_argument('--min_length', type=int)
	parser.add_argument('--sampling_rate', type=int)
	parser.add_argument('--mixture_data_list', type=str)
	args = parser.parse_args()
	
	main(args)