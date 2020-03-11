# Title: Super Resolution of images using SRGAN

#### Authors : Srikanth Babu Mandru

#### *Proposed as "Quantiphi Project"*

## Summary 

problem to solve and description of datasets- 

Super-resolution (SR) of images refers to process of generating or reconstructing the high-resolution (HR) images from low-resolution images (LR). This project mainly focuses on dealing with this problem of super-resolution using generative adversarial network, named SRGAN, a deep learning framework. In this project, SRGAN is trained and tested using the 'DIV2K' and 'MS-COCO' datasets [[6]](6) (specifically, 2017-validation images dataset) which are popular datasets for image resolution tasks. Images from both the datasets were merged and now data consists of:
1. 5800 training images (800 images from DIV2K and 5K images from MS-COCO)
2. 100 validation images (100 images from DIV2K validation set)

Non-technical description of methods


## Proposed plan of research

Phase 1 accomplishment brief - 
In first phase of this project, I have implemented the SRGAN which is a GAN-based model using TensorFlow, Keras and other Machine learning APIs. I choose Peak signal-to-noise-ratio [PSNR](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio) as the key metric to evaluate the model's performance. Introduced a new loss, namely 'Per-Pix' loss, for SRGAN model training and observed significant improvement in PSNR values with fewer iterations of training compared to model trained with 'Perceptual Loss'. 

Need for metric, loss and architecture:
 While doing research, I found the paper [8] which detailed about different metrics that can be used for evaluating the image resolution quality.


phase 2 plan - 
Now, in second phase of this project, I will focus on comparing the models trained between 'Per-Pix', 'Perceptual', 'MSE' losses through 'PSNR'. Apart from this, I will do research on various other models. Tensorboard will be implemented as part of visualisation tool, which is sophisticated and better to interpret the results while training.

It requires a huge effort to train this massive models. Thus, all of the training will be done using the Google cloud platform (GCP) AI services and products. During training, I make use of GPU (CUDA , cuDNN) toolkits to leverage faster training. After the training stage, the model will be deployed using google cloud AI platform for future predictions. Further, I am planning to deploy the model as an application to real-world users using TensorFlow Lite.



## Preliminary Results

Initially, I have implemented most of the image preprocessing part of project so that images data fits to our model seamlessly. The steps that I have followed are as follows :

- The actual original images are of size (2048, 1080, 3)

- Cropped the original images to size (256, 256, 3)

- Now, discriminator input will be of shape (256, 256, 3) and generator input is (64, 64, 3) which is downsampled version of discriminator input using "bicubic" kernel with factor of "4"   

Some of the sample low and high resolution images that are obtained from preprocessing stage are as shown in below figure:

<img src ="downloaded images/image_preprocess/low_res1.png" width = "400" height = "400" /> <img src ="downloaded  images/image_preprocess/high_res1.png" width = "400" height = "400" /> 

After training for 6 epochs (each epoch means training once over the entire dataset), I have obtained the results of PSNR as follows:

<img src ="downloaded images/psnr_results/epochs_srganloss_result.png" width = "400" height = "400" /> <img src ="downloaded_images/image_preprocess/high_res1.png" width = "400" height = "400" /> 

<img src ="downloaded images/psnr_results/epochs_perpixloss_result.png" width = "400" height = "400" /> <img src ="downloaded_images/image_preprocess/high_res1.png" width = "400" height = "400" /> 

To understand the perceptual quality of images, I have used the low-resolution version of validation images to generate the super-resolution images using both the models trained with "Perceptual Loss" and "Per-Pix Loss". The original high-resolution validation images and their corresponding generated images are as follows:

<img src ="downloaded images/perception_results/validation_samples_highres.png" width = "400" height = "400" /> <img src ="downloaded_images/image_preprocess/high_res1.png" width = "400" height = "400" /> 


<img src ="downloaded images/perception_results/valid_images_400.png" width = "400" height = "400" /> <img src ="downloaded_images/image_preprocess/high_res1.png" width = "400" height = "400" /> 

<img src ="downloaded images/perception_results/valid_images_500.png" width = "400" height = "400" /> <img src ="downloaded_images/image_preprocess/high_res1.png" width = "400" height = "400" /> 


## References

[1] I.Goodfellow, J.Pouget-Abadie, M.Mirza, B.Xu, D.Warde-Farley, S. Ozair, A. Courville, and Y. Bengio. Generative adversarial nets. In Advances in Neural Information Processing Systems (NIPS), pages 2672–2680, 2014.

[2] C. Ledig, L. Theis, F. Husza ́r, J. Caballero, A. Cunningham, A. Acosta, A. P. Aitken, A. Tejani, J. Totz, Z. Wang et al., “Photo-realistic single image super-resolution using a generative adversarial network,” in CVPR, 2017. [(SRGAN)](https://arxiv.org/abs/1609.04802)

[3] Zhihao Wang, Jian Chen, Steven C.H. Hoi, Fellow, "Deep Learning for Image Super-resolution: A Survey", IEEE, 2020.

[4] D. Kingma and J. Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations (ICLR), 2015.

[5] Nao Takano and Gita Alaghband. “SRGAN: Training Dataset Matters”, 2019. ( arXiv:1903.09922 ).

[6] Datasets Link:
1. DIV2K Dataset [Dataset-link](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
2. MS-COCO Dataset [Dataset-link](http://cocodataset.org/#download)

[7] Agustsson, Eirikur and Timofte, Radu. “NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study”, The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, July 2017.

[8] C.Y. Yang, C. Ma, and M.H. Yang. Single-image super-resolution:A benchmark. In European Conference on Computer Vision (ECCV),pages 372–386. Springer, 2014.


**Advantages of Super-Resolution:**
 
- It saves the storage space of images and provides high resolution images whenever needed

- Adapts to new hardware upgrades(like improved screen resolution of TV, Theatre, etc)

- Make objects to be highly distinguishable in images so that data in whole will be useful for other computer vision tasks.
