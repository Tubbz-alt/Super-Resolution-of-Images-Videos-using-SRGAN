# Title: Super Resolution of images using SRGAN

## Authors : Srikanth Babu Mandru

## Summary 

Problem wish to solve :

Data being the modern world's electricity, it's driving as fuel in many applications ranging from Medical, Industries, businesses. One of major issues with systems depending on data is 'Data' itself. There might be a lot of descrepencies in the quality of data available to use. This is true, especially in computer vision field, where the images or videos available to use are not upto the requirement and resulting in less efficiencies of the systems. As the data and technology evolves, there is need of technique that acts as fast fuel for improving the efficiencies of existing or other techniques used for different appilcations. For all the computer vision related problems, images or videos that we input makes a huge deal in alleviating the overall performance. It is highly challenging task to estimate a high-resolution (HR) image from its low-resolution (LR) counterpart. It is often referred to as an 'Super-Resolution' (SR). SR received substantial attention from within the computer vision research community and has a wide range of applications.


description of proposed methods and datasets:

Initially, the project aims at implementing the SRGAN model to solve Super Resolution(SR) problem. I have taken dataset named 'DIV2K' [[Dataset link]](https://data.vision.ee.ethz.ch/cvl/DIV2K/), which is one of the popular datasets for image resolution task. Later, I try to incorporate other datasets for training so that model become more robust. After dealing with images, I make this entire model to work for videos based on the availability of datasets.


Applications :

TV resolution or movie resolution,
Face Recognition,
Autonomous vehicle driving,
Medical, 
Security Systems,
Gaming and Graphics design.

## Proposed plan of research

details of methods : like processing , visualization, ml methods and statistical methods


This project focuses on solving the SR problem through 2 phases. First phase of project is mainly exploring the ideas of GAN architectures and understanding the merits of using SRGAN. At this phase, I will also concentrate developing SRGAN model using the Tensorflow, Keras and other Machine learning APIs. In the second phase, I will take the training of model to google cloud platform using the google storage and compute engines. Over the entire project cycle, I do some research on building other architectures for improving the performance of SR with strong focus on Deep learning techinques. 

As a metric to evalute, there are two options available, namely 'Peak-Signal-to-Noise-Ratio (PSNR)' and Mean-Opinion-Score (MOS). While the 'PSNR' is mostly used for evaluation in image processing, it does not correlate with the perceptual quality of image and not able to capture the essence of perception that is more obvious for human to interprete. So, I am planning to take 'MOS' as metric (which is similar to giving ratings). This way, we could get the better super resolved images. 

GAN architecture photo: 

<img src ="https://developers.google.com/machine-learning/gan/images/gan_diagram.svg" width = "1000" height = "200" >


## Preliminary Results

Summary statistics 

Image preprocessing : 
steps : 
- The original images are of size (2048, 1080, 3).

-Cropped the original images to size (256, 256, 3) ( different from shape in SRGAN paper)

- Now, discriminator input will be of shape (256, 256, 3) and generator input is (96, 96, 3) which is downsampled using "bicubic" kernel with factor of "4".   

Images of low res and high res. 



## References

[1] C. Ledig, L. Theis, F. Husza ́r, J. Caballero, A. Cunningham, A. Acosta, A. P. Aitken, A. Tejani, J. Totz, Z. Wang et al., “Photo-realistic single image super-resolution using a generative adversarial network,” in CVPR, 2017. [(SRGAN)](https://arxiv.org/abs/1609.04802)


[2] Zhihao Wang, Jian Chen, Steven C.H. Hoi, Fellow, "Deep Learning for Image Super-resolution: A Survey", IEEE, 2019.

