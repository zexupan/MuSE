import torch
import os
from tqdm import tqdm
import numpy as np
import argparse
import cv2 as cv

import sys
sys.path.append('../../')
from pretrain_networks.visual_frontend import VisualFrontend

def preprocess_sample(file, params, args):

    """
    Function to preprocess each data sample.
    """

    videoFile = args.video_data_direc + file + ".mp4"
    roiFile = args.visual_frame_direc + file +".png"
    visualFeaturesFile = args.lip_embedding_direc + file + ".npy"

    if os.path.exists(visualFeaturesFile):
    	return

    if not os.path.exists(roiFile[:-9]):
        os.makedirs(roiFile[:-9])

    if not os.path.exists(visualFeaturesFile[:-9]):
        os.makedirs(visualFeaturesFile[:-9])


    roiSize = params["roiSize"]
    normMean = params["normMean"]
    normStd = params["normStd"]
    vf = params["vf"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    #for each frame, resize to 224x224 and crop the central 112x112 region
    captureObj = cv.VideoCapture(videoFile)
    roiSequence = list()
    while (captureObj.isOpened()):
        ret, frame = captureObj.read()
        if ret == True:
            grayed = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            grayed = grayed/255
            grayed = cv.resize(grayed, (roiSize*2,roiSize*2))
            roi = grayed[int(roiSize-(roiSize/2)):int(roiSize+(roiSize/2)), int(roiSize-(roiSize/2)):int(roiSize+(roiSize/2))]
            roiSequence.append(roi)
        else:
            break
    captureObj.release()
    # cv.imwrite(roiFile, np.floor(255*np.concatenate(roiSequence, axis=1)).astype(np.int))


    #normalise the frames and extract features for each frame using the visual frontend
    #save the visual features to a .npy file
    inp = np.stack(roiSequence, axis=0)
    inp = np.expand_dims(inp, axis=[1,2])
    inp = (inp - normMean)/normStd
    inputBatch = torch.from_numpy(inp)
    inputBatch = (inputBatch.float()).to(device)
    vf.eval()
    with torch.no_grad():
        outputBatch = vf(inputBatch)
    out = torch.squeeze(outputBatch, dim=1)
    out = out.cpu().numpy()
    np.save(visualFeaturesFile, out)
    return

def main(args):
    gpuAvailable = torch.cuda.is_available()
    device = torch.device("cuda" if gpuAvailable else "cpu")

    #declaring the visual frontend module
    vf = VisualFrontend()
    vf.load_state_dict(torch.load('../../pretrain_networks/visual_frontend.pt', map_location=device))
    vf.to(device)

	#walking through the data directory and obtaining a list of all files in the dataset
    filesList = list()
    for root, dirs, files in os.walk(args.video_data_direc+'train/'):
        for file in files:
            if file.endswith(".mp4"):
                path = root.split('/')
                filesList.append((path[-3]+'/'+path[-2]+'/'+path[-1]+'/'+file[:-4]))
    print(len(filesList))

    params = {"roiSize":112, "normMean":0.4161, "normStd":0.1688, "vf":vf}
    for file in tqdm(filesList, leave=True, desc="Preprocess", ncols=75):
        preprocess_sample(file, params, args)



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='LRS3 dataset')
	parser.add_argument('--video_data_direc', type=str)
	parser.add_argument('--lip_embedding_direc', type=str)
	parser.add_argument('--visual_frame_direc', type=str)
	args = parser.parse_args()
	main(args)