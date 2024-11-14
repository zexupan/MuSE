## MuSE

A PyTorch implementation of the [Muse: Multi-modal target speaker extraction with visual cues](https://arxiv.org/abs/2010.07775)

## Update:
* A new version of this code is scheduled to be released [here (ClearVoice repo)](https://github.com/modelscope/ClearVoice). 
* The dataset can be found [here](https://huggingface.co/datasets/alibabasglab/KUL-mix).
  
## Project Structure

`/data/voxceleb2-800`: Scripts to preprocess the voxceleb2 datasets.

`/pretrain_networks`: The visual front-end network

`/src`: The training scripts

## Pre-trained Weights
Download the pre-trained weights for the Visual Frontend and place it in the ./pretrain_networks folder using the following command:

	wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1k0Zk90ASft89-xAEUbu5CmZWih_u_lRN' -O visual_frontend.pt


## References
1. The pre-trained weights of the Visual Frontend have been obtained from [Afouras T. and Chung J, Deep Audio-Visual Speech Recognition](https://github.com/lordmartian/deep_avsr) GitHub repository.

2. The model is adapted from [Conv-TasNet](https://github.com/kaituoxu/Conv-TasNet) GitHub repository.
