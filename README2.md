# Title: Super Resolution of images using SRGAN

#### Authors : Srikanth Babu Mandru

#### *Proposed as "Quantiphi Project"*

## Summary 

Super-resolution (SR) of images refers to process of generating or reconstructing the high- resolution (HR) images from low-resolution images (LR). This project mainly focuses on dealing with this problem of super-resolution using generative adversarial network, named SRGAN, a deep learning framework. In this project, SRGAN is trained and tested using the 'DIV2K' and MS-COCO datasets which are popular datasets for image resolution tasks. Placed 
 dataset consists of:
1. 800 training
2. 100 validation images
Most of this project is built upon the ideas of
SRGAN paper [2]. Apart from that, I did some research on comparing the results obtained using different objective functions available in TensorFlow’s “tfgan” library for loss optimizations of SRGAN. Different model implementations are evaluated through the peak signal-to-noise ratio (PSNR) scores as metric. Intuitively, this metric does not capture the essence of perceptual quality of image. However, it is comparatively easy to use PSNR when evaluating the performance while training the model compared to mean-opinion-score (MOS) that has been used by authors of paper [2]. This paper also proposes a method of super-resolution using SRGAN with “Per-Pix loss” which I defined in the losses section of this paper. Based on results from [2] and [5], I have combined both MSE and VGG losses, named it as “Per-Pix loss” that stands for ‘Perceptual and Pixel’ qualities of image, which resulted in decent PSNR scores with less iterations of training. Finally, achieved the PSNR value peaked at approximately “14.5” and “14.05” on training and validation datasets respectively.

Datasets: 
1. DIV2K Dataset [Dataset-link](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
2. MS-COCO Dataset [Dataset-link](http://cocodataset.org/#download)


## Proposed plan of research

This project focuses on solving the SR problem through 2 phases. First phase of project is to explore the ideas of GAN architectures and to understand the merits of using SRGAN. At this phase, I will also concentrate on developing SRGAN model using the Tensorflow, Keras and other Machine learning APIs. In the second phase, I will take the training of model to google cloud platform using the google storage and compute engines. Over the entire project cycle, I will do some research on building other architectures for improving the performance of SR by strongly focusing on Deep learning techinques. 

As a metric to evalute, there are two options available, namely 'Peak-Signal-to-Noise-Ratio (PSNR)' and 'Mean-Opinion-Score (MOS)'. While the 'PSNR' is mostly used for evaluation in image processing, it does not correlate with the perceptual quality of image and does not capture the essence of perception that is more obvious for human to interpret. So, I will consider to take 'MOS' as the evaluation metric (which is similar to giving ratings). This way, we could get better super resolved images. 

General GAN block diagram is shown in following figure: 

<img src="https://developers.google.com/machine-learning/gan/images/gan_diagram.svg" width = "1000" height = "300" />

Generator and discriminator network architectures in SRGAN are as shown in below figure:

<img src="https://www.oreilly.com/library/view/generative-adversarial-networks/9781789136678/assets/fe3eced0-c452-4b7a-9f6d-dd8228048ab9.png" width = "1000" height = "500" />

## Preliminary Results

As the preliminary steps, I have implemented most of the image preprocessing part of project so that images data fits to our model seamlessly. The steps that I have followed are as follows :

- The actual original images are of size (2048, 1080, 3)

- Cropped the original images to size (256, 256, 3) ( different from shape in SRGAN paper)

- Now, discriminator input will be of shape (256, 256, 3) and generator input is (64, 64, 3) which is downsampled version of discriminator input using "bicubic" kernel with factor of "4"   

Some of the sample low and high resolution images that are obtained from preprocessing steps are as shown in below figure:

<img src ="example_images/lowres/img2.png" width = "400" height = "400" /> <img src ="example_images/highres/img_2.png" width = "400" height = "400" /> 

<img src ="example_images/lowres/img3.png" width = "400" height = "400" /> <img src ="example_images/highres/img_3.png" width = "400" height = "400" /> 

<img src ="example_images/lowres/img10.png" width = "400" height = "400" /> <img src ="example_images/highres/img_10.png" width = "400" height = "400" />

## References

[1] C. Ledig, L. Theis, F. Husza ́r, J. Caballero, A. Cunningham, A. Acosta, A. P. Aitken, A. Tejani, J. Totz, Z. Wang et al., “Photo-realistic single image super-resolution using a generative adversarial network,” in CVPR, 2017. [(SRGAN)](https://arxiv.org/abs/1609.04802)


[2] Zhihao Wang, Jian Chen, Steven C.H. Hoi, Fellow, "Deep Learning for Image Super-resolution: A Survey", IEEE, 2019.

**Advantages of Super-Resolution:**
 
- It saves the storage space of images and provides high resolution images whenever needed

- Adapts to new hardware upgrades(like improved screen resolution of TV, Theatre, etc)

- Make objects to be highly distinguishable in images so that data in whole will be useful for our tasks.
