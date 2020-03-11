# Title: Super Resolution of images using SRGAN

#### Authors : Srikanth Babu Mandru

#### *Proposed as "Quantiphi Project"*

## Summary 

Super-resolution (SR) of images refers to process of generating or reconstructing the high-resolution (HR) images from low-resolution images (LR). This project mainly focuses on dealing with this problem of super-resolution using generative adversarial network, named SRGAN, a deep learning framework. In this project, SRGAN is trained and tested using the 'DIV2K' and 'MS-COCO' datasets [6] (specifically, 2017-validation images dataset) which are popular datasets for image resolution tasks. Images from both the datasets were merged and now data consists of:
1. 5800 training images (800 images from DIV2K and 5K images from MS-COCO)
2. 100 validation images (100 images from DIV2K validation set)

Most of this project is built upon the ideas of SRGAN paper [2] which showed better super-resolution of images in terms of perceptual quality compared to other deep learning models. Based on results from [2] and [5], I have combined both MSE and VGG losses, named it as “Per-Pix loss” that stands for ‘Perceptual and Pixel’ qualities of image, which resulted in decent PSNR scores with less iterations of training. At present, in phase 2 of this project, I will train the models that I have implemented in phase 1 using the Google cloud AI platform and compare the performance between those models. Later, I deploy the model on cloud and try to create an application for real-world users. Finally, if it is feasible, I will super-resolve the videos [6] (video dataset consists of 4 videos) using the best model.

## Proposed plan of research

In first phase of this project, I have implemented the SRGAN which is a GAN-based model using TensorFlow, Keras and other Machine learning APIs. I choose Peak signal-to-noise-ratio [PSNR](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio) as the key metric to evaluate the model's performance. Proposed a new loss, namely 'Per-Pix' loss, for SRGAN model training and observed significant improvement in PSNR values with fewer iterations of training compared to model trained with 'Perceptual Loss'.

Now, in second phase of this project, I pickup from the first phase results and focus on comparing the model performances trained separately with 'Per-Pix', 'Perceptual', 'MSE' losses through 'PSNR' metric. Apart from this, I will do research on using various other model architectures. There is also a great need for proper metric to evaluate the image quality. For this, currently, I found the paper [8] which detailed about different metrics that can be used for evaluating the image resolution quality. In this paper, they have described how various metrics are related to the perceptual quality. So, I will study further on papers [3] and [8] to get deeper understanding and arrive at right approaches in order to solve super-resolution problem. If I find any reasonable approach or ideas that would impact the performance, I will incorporate those into the project.

Coming to training stage of this project, it requires a huge effort to train these massive models. Thus, all of the training will be done using the Google cloud platform (GCP) AI services and products. During training, I make use of NVIDIA Tesla P100 GPUs with CUDA and cuDNN toolkits to leverage faster training offered by GPUs. As a part of training procedure, I will also create a visualization dashboard consisting of model architecture and results using TensorBoard to better interpret the results while training is in progress. After the training stage, the model will be deployed using google cloud AI platform for future predictions. Also, the best model will be used to super-resolve the videos using a new data pipeline to process the videos. Further, I am planning to deploy the model as an application to real-world users using TensorFlow Lite. Overall, in phase 2, I primarily concentrate on training and deploying the SRGAN model besides doing further research.


## Preliminary Results

Initially, I have implemented most of the image preprocessing part of project so that images data fits to our model seamlessly. The steps that I have followed are as follows :

- The actual original images are of size (2048, 1080, 3)

- Cropped the original images to size (256, 256, 3)

- Now, discriminator input will be of shape (256, 256, 3) and generator input is (64, 64, 3) which is downsampled version of discriminator input using "bicubic" kernel with factor of "4"   

Some of the sample low and high resolution images that are obtained from image preprocessing stage are as shown in below figure:

<img src ="downloaded images/image_preprocess/low_res1.png" width = "400" height = "400" /> <img src ="downloaded images/image_preprocess/high_res1.png" width = "400" height = "400" /> 

By the end of phase 1, after training the models for 6 epochs (each epoch means training once over the entire dataset), I have obtained the results of PSNR as shown in below figure. From the figure, it can be inferred that model trained with "Perceptual loss" did not reach good PSNR value as seen from the left plot in below figure. On the other hand, model trained with "Per-Pix" loss has started showing good results in terms of PSNR from the first epoch itself (seen from right plot in below figure).

<img src ="downloaded images/psnr_results/epochs_srganloss_result.png" width = "400" height = "400" /> <img src ="downloaded images/psnr_results/epochs_perpixloss_result.png" width = "400" height = "400" /> 

To understand the perceptual quality of images, I have used the low-resolution version of validation images to generate the super-resolution images using both the models trained with "Perceptual Loss" and "Per-Pix Loss". From the figures below, we can infer that "Per-Pix" loss is pushing the model performance significantly resulting in obtaining the images close to the real images. The real high-resolution validation images and their corresponding generated images from both models trained with "Perceptual loss" and "Per-Pix loss" are as shown in below figures:

<img src ="downloaded images/perception_results/validation_samples_highres.png" width = "800" height = "400" /> 


<img src ="downloaded images/perception_results/valid_images_400.png" width = "800" height = "400" />


<img src ="downloaded images/perception_results/valid_images_500.png" width = "800" height = "400" /> 


## References

[1] I.Goodfellow, J.Pouget-Abadie, M.Mirza, B.Xu, D.Warde-Farley, S. Ozair, A. Courville, and Y. Bengio. Generative adversarial nets. In Advances in Neural Information Processing Systems (NIPS), pages 2672–2680, 2014.

[2] C. Ledig, L. Theis, F. Husza ́r, J. Caballero, A. Cunningham, A. Acosta, A. P. Aitken, A. Tejani, J. Totz, Z. Wang et al., “Photo-realistic single image super-resolution using a generative adversarial network,” in CVPR, 2017. [(SRGAN)](https://arxiv.org/abs/1609.04802)

[3] Zhihao Wang, Jian Chen, Steven C.H. Hoi, Fellow, "Deep Learning for Image Super-resolution: A Survey", IEEE, 2020.

[4] D. Kingma and J. Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations (ICLR), 2015.

[5] Nao Takano and Gita Alaghband. “SRGAN: Training Dataset Matters”, 2019. ( arXiv:1903.09922 ).

[6] Datasets Link:
1. DIV2K Dataset [Dataset-link](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
2. MS-COCO Dataset [Dataset-link](http://cocodataset.org/#download)
3. Vid4 Dataset [Dataset-link](https://xinntao.github.io/open-videorestoration/rst_src/datasets_sr.html)

[7] Agustsson, Eirikur and Timofte, Radu. “NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study”, The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, July 2017.

[8] C.Y. Yang, C. Ma, and M.H. Yang. Single-image super-resolution:A benchmark. In European Conference on Computer Vision (ECCV),pages 372–386. Springer, 2014.


**Advantages of Super-Resolution:**
 
- It saves the storage space of images and provides high resolution images whenever needed

- Adapts to new hardware upgrades(like improved screen resolution of TV, Theatre, etc)

- Make objects to be highly distinguishable in images so that data in whole will be useful for other computer vision tasks.
