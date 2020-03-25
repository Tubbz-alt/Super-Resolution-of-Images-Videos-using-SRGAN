
import tensorflow as tf
import matplotlib.pyplot as plt
import os

def preprocess_load(root_path, hr_image_shape = None , lr_image_shape = None, original_image_shape = None , show_images = False , data_path = None):
    
    discriminator_img_shape = hr_image_shape
    generator_img_shape = lr_image_shape
    original_img_shape =  original_image_shape

    def decode_img(img, hr_image_shape = (256, 256) , lr_image_shape = (64, 64), original_image_shape = (2048, 1080) ):

      IMG_WIDTH, IMG_HEIGHT = hr_image_shape
      # convert the compressed string to a 3D uint8 tensor
      img = tf.image.decode_jpeg(img, channels=3)

      # Use `convert_image_dtype` to convert to floats in the [0,1] range.
      img = tf.image.convert_image_dtype(img, tf.float32)
      
      # resize the image to the desired size.
      hr_img = tf.image.resize_with_crop_or_pad(img, IMG_HEIGHT , IMG_WIDTH) 

      # lr_img = tf.image.resize(img, [ int(original_img_shape[1]//4), int(original_img_shape[0]//4) ]  , method = 'bicubic' ) 
      # scipy.misc.imresize

      # lr_final_shape = np.min(int(original_img_shape[1]//4), int(original_img_shape[0]//4)) 

      lr_img = tf.image.resize(hr_img, [ int(lr_image_shape[1]), int(lr_image_shape[0]) ]  , method = 'bicubic' ) 

      return lr_img, hr_img

    def process_path(file_path, hr_image_shape = discriminator_img_shape , lr_image_shape = generator_img_shape, original_image_shape = original_img_shape ):
      # load the raw data from the file as a string
      img = tf.io.read_file(file_path)
      lr_img, hr_img = decode_img(img , hr_image_shape = hr_image_shape , lr_image_shape = lr_image_shape ,original_image_shape = original_image_shape )
      return lr_img , hr_img

    def _hr_preprocess(element):
      # Map [0, 255] to [-1, 1].
      # images = (tf.cast(element, tf.float32) - 127.5) / 127.5
      images = (tf.cast(element, tf.float32) * 2.0 ) - 1.0
      return images

    def _lr_preprocess(element):
      # Map [0, 255] to [0, 1].
      # images = (tf.cast(element, tf.float32)) / 255.0
      images = (tf.cast(element, tf.float32))
      return images

    def show_hr_image(image, label = 'high resolution image example', img_ids_hr = 1):
      # image = (tf.cast(image, tf.float32) * 127.5 ) + 127.5
      image = (tf.cast(image, tf.float32) + 1.0 ) / 2.0
      plt.figure(figsize=(10, 10))
      plt.imshow(image)
      plt.title(label)
      plt.axis('off')
      plt.savefig(root_path + "example_images/highres/img_{}.png" .format(img_ids_hr ) )


    def show_lr_image(image, label = 'low resolution image example', img_ids_lr = 1):
      # image = (tf.cast(image, tf.float32) * 255.0 ) 
      image = (tf.cast(image, tf.float32) )
      plt.figure(figsize=(10, 10))
      plt.imshow(image)
      plt.title(label)
      plt.axis('off')
      plt.savefig(root_path + "example_images/lowres/img{}.png" .format(img_ids_lr ) )


  # Start loading from the directory using above defined functions

    list_ds = tf.data.Dataset.list_files(str(root_path + data_path), shuffle = False)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    images_ds = list_ds.map(process_path, num_parallel_calls= AUTOTUNE)

  # Get the individual datasets
    lr_img_ds = images_ds.interleave(lambda lr_img,hr_img: tf.data.Dataset.from_tensors(lr_img))
    hr_img_ds = images_ds.interleave(lambda lr_img,hr_img: tf.data.Dataset.from_tensors(hr_img))

  # modify range of values
    highres_images = hr_img_ds.map(_hr_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    lowres_images = lr_img_ds.map(_lr_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if show_images == True:
      ### show high resolution images 
      for i, image1 in enumerate(highres_images.take(5)):  
        show_hr_image(image1, img_ids_hr = i+1)
      
      ### show low resolution images
      for j,image2 in enumerate(lowres_images.take(5)):  
        show_lr_image(image2, img_ids_lr= j + 1 )

    return tf.data.Dataset.zip((lowres_images, highres_images)) 
