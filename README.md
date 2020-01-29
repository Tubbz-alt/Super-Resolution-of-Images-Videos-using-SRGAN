# Title: Super Resolution of images using SRGAN

#### Authors : Srikanth Babu Mandru

#### *Proposed as "Quantiphi Project"*

## Summary 

Problem wish to solve :

Data being like the modern world's electricity, it's driving as fuel in many applications ranging from Medical, Industries, Businesses. One of major issues with systems depending on data is 'Data' itself. There might be a lot of discrepancies in the quality of data available to use. This is true, especially in computer vision field, where the images or videos available for the use are not upto the requirement and resulting in less efficiencies of the systems. As the data and technology evolves, there is need of technique that make this act as a fast fuel for improving the efficiencies of existing or other techniques used for different appilcations. For all the computer vision related problems, the images or videos that we feed into systems makes a huge deal in alleviating the overall performance. Thus, it is essential to make the images or videos more accurate enough for our tasks. It is highly challenging task to estimate a high-resolution (HR) image from its low-resolution (LR) counterpart and this process of convertion is often referred to as an 'Super-Resolution' (SR) problem. 'Super Resolution' received huge attention from within the computer vision research community in the recent years and has a wide range of applications such as TV resolution or movie resolution, Face Recognition or verification, Autonomous vehicle driving, Medical Imaging,  Security Systems, Gaming and Graphics design, etc.


description of proposed methods and datasets:

Initially, the project aims at implementing the SRGAN model to solve Super Resolution(SR) problem. I have taken dataset named 'DIV2K' [[Dataset link]](https://data.vision.ee.ethz.ch/cvl/DIV2K/), which is one of the popular datasets for image resolution task. 'DIV2k' dataset consititute of 800 training, 100 validation, 100 test images .Later, I try to incorporate other datasets for training so that model become more robust. After dealing with images, I make transmission to work with videos based on the availability of datasets.


Advantages :
 
storage of images by processing while needed

Adapt to new hardware (like improved screen resolution of TV, Theatre, etc)

Make objects to be better distinguishable in images


## Proposed plan of research

details of methods : like processing , visualization, ml methods and statistical methods


This project focuses on solving the SR problem through 2 phases. First phase of project is mainly exploring the ideas of GAN architectures and understanding the merits of using SRGAN. At this phase, I will also concentrate developing SRGAN model using the Tensorflow, Keras and other Machine learning APIs. In the second phase, I will take the training of model to google cloud platform using the google storage and compute engines. Over the entire project cycle, I do some research on building other architectures for improving the performance of SR with strong focus on Deep learning techinques. 

As a metric to evalute, there are two options available, namely 'Peak-Signal-to-Noise-Ratio (PSNR)' and Mean-Opinion-Score (MOS). While the 'PSNR' is mostly used for evaluation in image processing, it does not correlate with the perceptual quality of image and not able to capture the essence of perception that is more obvious for human to interprete. So, I am planning to take 'MOS' as metric (which is similar to giving ratings). This way, we could get the better super resolved images. 

GAN architecture photo: 

<img src ="https://developers.google.com/machine-learning/gan/images/gan_diagram.svg" width = "1000" height = "200" />


## Preliminary Results

Summary statistics 

Image preprocessing : 
steps : 
- The actual original images are of size (2048, 1080, 3).

-Cropped the original images to size (256, 256, 3) ( different from shape in SRGAN paper)

- Now, discriminator input will be of shape (256, 256, 3) and generator input is (96, 96, 3) which is downsampled using "bicubic" kernel with factor of "4".   

Sample of high resolution and low resolution images are obtained and as shown in below figure:

<img src ="example_images/lowres/img2.png" width = "400" height = "400" /> <img src ="example_images/highres/img_2.png" width = "400" height = "400" /> 

<img src ="example_images/lowres/img3.png" width = "400" height = "400" /> <img src ="example_images/highres/img_3.png" width = "400" height = "400" /> 

<img src ="example_images/lowres/img10.png" width = "400" height = "400" /> <img src ="example_images/highres/img_10.png" width = "400" height = "400" />

## References

[1] C. Ledig, L. Theis, F. Husza ́r, J. Caballero, A. Cunningham, A. Acosta, A. P. Aitken, A. Tejani, J. Totz, Z. Wang et al., “Photo-realistic single image super-resolution using a generative adversarial network,” in CVPR, 2017. [(SRGAN)](https://arxiv.org/abs/1609.04802)


[2] Zhihao Wang, Jian Chen, Steven C.H. Hoi, Fellow, "Deep Learning for Image Super-resolution: A Survey", IEEE, 2019.

