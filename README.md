# BIlinear-Network-for-Dehazing
This is the matlab code for bilinear network using composition loss for dehazing.

Training data preparation
We use the NYU2 dataset. Download them from website "http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html". 
Use "gnerate_hazy_img_noise.m" to generate hazy noise images.
Use "gnerate_hazy_img_nyu.m" to generate hazy noise-free images.

Then use "generate_train.m" to make training data. Note "folder", "folderhazy" and "folderdepth" are for clear ground truth images, hazy images, depth maps respectively. Change them to your own path.

Training
use train.m to begin training. 

Loss function
vl_nnhazerobustloss.m  ---> The L2 norm loss used in the paper.
vl_nnhazesquareloss_non_noise.m  ---> The L2 norm loss used for noise-free training in the paper.

Testing
Use "demo_test.m" to see the dehazed and denoised results of trained models.
