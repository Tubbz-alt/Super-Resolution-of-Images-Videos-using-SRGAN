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
In first phase of this project, I have implemented the SRGAN model using TensorFlow, Keras and other Machine learning APIs. I took Peak signal-to-noise-ratio [PSNR](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio) as the key metric to evaluate the model's performance. Introduced the new loss, namely 'Per-Pix' loss, for SRGAN model training and observed significant improvement in PSNR values with fewer iterations of training compared to model trained with 'Perceptual Loss'. 


phase 2 plan - 
Now, in second phase of this project, I will focus on comparing the models trained between 'Per-Pix', 'Perceptual', 'MSE' losses. 

All of the training will be done using the Google cloud platform (GCP) AI services and products.

Google cloud platform training



## Preliminary Results

Initially, I have implemented most of the image preprocessing part of project so that images data fits to our model seamlessly. The steps that I have followed are as follows :

- The actual original images are of size (2048, 1080, 3)

- Cropped the original images to size (256, 256, 3)

- Now, discriminator input will be of shape (256, 256, 3) and generator input is (64, 64, 3) which is downsampled version of discriminator input using "bicubic" kernel with factor of "4"   

Some of the sample low and high resolution images that are obtained from preprocessing stage are as shown in below figure:

<img src ="example_images/lowres/img2.png" width = "400" height = "400" /> <img src ="example_images/highres/img_2.png" width = "400" height = "400" /> 


## References

[1] I.Goodfellow, J.Pouget-Abadie, M.Mirza, B.Xu, D.Warde-Farley, S. Ozair, A. Courville, and Y. Bengio. Generative adversarial nets. In Advances in Neural Information Processing Systems (NIPS), pages 2672–2680, 2014.

[2] C. Ledig, L. Theis, F. Husza ́r, J. Caballero, A. Cunningham, A. Acosta, A. P. Aitken, A. Tejani, J. Totz, Z. Wang et al., “Photo-realistic single image super-resolution using a generative adversarial network,” in CVPR, 2017. [(SRGAN)](https://arxiv.org/abs/1609.04802)

[3] Zhihao Wang, Jian Chen, Steven C.H. Hoi, Fellow, "Deep Learning for Image Super-resolution: A Survey", IEEE, 2019.

[4] D. Kingma and J. Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations (ICLR), 2015.

[5] Nao Takano and Gita Alaghband. “SRGAN: Training Dataset Matters”, 2019. ( arXiv:1903.09922 ).

[6] Datasets Link:
1. DIV2K Dataset [Dataset-link](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
2. MS-COCO Dataset [Dataset-link](http://cocodataset.org/#download)

[7] Agustsson, Eirikur and Timofte, Radu. “NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study”, The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, July 2017.



**Advantages of Super-Resolution:**
 
- It saves the storage space of images and provides high resolution images whenever needed

- Adapts to new hardware upgrades(like improved screen resolution of TV, Theatre, etc)

- Make objects to be highly distinguishable in images so that data in whole will be useful for other computer vision tasks.
