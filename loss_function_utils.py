

import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

tf.keras.backend.set_image_data_format('channels_last')

## For distributed training
from tensorflow.python.keras.utils import losses_utils

GLOBAL_BATCH_SIZE = None

def compute_avg_loss_gpu(per_replica_loss_tensor , GLOBAL_BATCH_SIZE_IN = GLOBAL_BATCH_SIZE):
    return tf.nn.compute_average_loss(per_replica_loss_tensor, global_batch_size = GLOBAL_BATCH_SIZE_IN )

def vgg_model():
  # Load our model. Load pretrained VGG, trained on imagenet data
  with tf.name_scope("VGG_MODEL_LOADED_BLOCK"):
    vgg = tf.keras.applications.VGG19(input_shape = [None, None, 3] , include_top=False, weights='imagenet')
    vgg.trainable = False
    
    outputs = [vgg.get_layer(name).output for name in ['block5_conv4']]

    model = tf.keras.models.Model(vgg.input, outputs)
    model.trainable = False 
  return model
  

content_layers = ['block5_conv4']

def vgg_layers(input_tensor):
  """ Creates a vgg model that returns a list of intermediate output values."""
  # Load our model. Load pretrained VGG, trained on imagenet data

  with tf.name_scope("GAN_TO_VGG_BLOCK"):
    vgg_model_loaded = vgg_model()

    output_features = vgg_model_loaded(input_tensor)

  return tf.convert_to_tensor(output_features, dtype = tf.float32)
  
  

def content_loss(hr, sr):
    with tf.name_scope("Content_Loss_Block"):
          # sr_preprocessed = preprocess_input(sr)
          # hr_preprocessed = preprocess_input(hr)
          sr_features = vgg_layers(sr) / 12.75
          hr_features = vgg_layers(hr) / 12.75
          meansquarederror_obj = MeanSquaredError(reduction=losses_utils.ReductionV2.NONE)
          per_replica_loss = meansquarederror_obj(hr_features, sr_features)
          meansquarederror_loss = compute_avg_loss_gpu(per_replica_loss )
          return meansquarederror_loss

def generator_loss(sr_out):
    with tf.name_scope("Generator_Loss_Block"):       
        binaryCrossentropy_obj = BinaryCrossentropy(from_logits=False, reduction = losses_utils.ReductionV2.NONE )

        binaryCrossentropy_loss_per_replica = binaryCrossentropy_obj(tf.ones_like(sr_out), sr_out)

        binaryCrossentropy_loss = compute_avg_loss_gpu(binaryCrossentropy_loss_per_replica )

        return binaryCrossentropy_loss

def discriminator_loss(gan_model):
    with tf.name_scope("Discriminator_Loss_Block"):
        hr_out = tf.cast(gan_model.discriminator_real_outputs, tf.float32)
        sr_out = tf.cast(gan_model.discriminator_gen_outputs, tf.float32)

        def get_loss_over_batch(x , y ):
            binaryCrossentropy_obj_disc = BinaryCrossentropy(from_logits=False, reduction = losses_utils.ReductionV2.NONE)
            binaryCrossentropy_per_replica = binaryCrossentropy_obj_disc(x, y)
            return compute_avg_loss_gpu(binaryCrossentropy_per_replica )

        hr_loss = get_loss_over_batch(tf.ones_like(hr_out), hr_out)

        sr_loss = get_loss_over_batch(tf.zeros_like(sr_out), sr_out)

        return hr_loss + sr_loss

def srgan_loss(gan_model):
    with tf.name_scope("SRGAN_Loss_Block"):
        # Define non-adversarial loss - for example "VGG loss" and "MSE loss"

        non_adversarial_loss = content_loss(gan_model.real_data, gan_model.generated_data )

        # non_adversarial_loss_MSE = tf.keras.losses.mean_squared_error(gan_model.real_data, gan_model.generated_data) 

        # Define generator loss
        generator_loss_value =  0.001 * generator_loss(gan_model.discriminator_gen_outputs )
        # generator_loss = tfgan.losses.modified_generator_loss(gan_model)

        # Combine these losses - you can specify more parameters
        # Exactly one of weight_factor and gradient_ratio must be non-None

        combined_loss = non_adversarial_loss + generator_loss_value

    return combined_loss
    
    
def perpix_loss(gan_model):
    with tf.name_scope("PERPIX_Loss_Block"):
        
        # Define non-adversarial loss - for example "VGG loss" and "MSE loss"

        non_adversarial_loss = content_loss(gan_model.real_data, gan_model.generated_data)

        meansquarederror_obj2 = MeanSquaredError(reduction=losses_utils.ReductionV2.NONE)
        per_replica_loss2 = meansquarederror_obj2(gan_model.real_data, gan_model.generated_data)
        non_adversarial_loss_MSE = compute_avg_loss_gpu(per_replica_loss2 )

        # Define generator loss
        generator_loss_value =  0.001 * generator_loss(gan_model.discriminator_gen_outputs )
        # generator_loss = tfgan.losses.modified_generator_loss(gan_model)

        # Combine these losses - you can specify more parameters
        # Exactly one of weight_factor and gradient_ratio must be non-None

        combined_loss = non_adversarial_loss + non_adversarial_loss_MSE + generator_loss_value


    return combined_loss
    
def MSE_loss(gan_model):
    with tf.name_scope("MSE_Loss_Block"):
        
        # Define non-adversarial loss - for example "VGG loss" and "MSE loss"

        meansquarederror_obj2 = MeanSquaredError(reduction=losses_utils.ReductionV2.NONE)
        per_replica_loss2 = meansquarederror_obj2(gan_model.real_data, gan_model.generated_data)
        non_adversarial_loss_MSE = compute_avg_loss_gpu(per_replica_loss2 )

        # Define generator loss
        generator_loss_value =  0.001 * generator_loss(gan_model.discriminator_gen_outputs )
        # generator_loss = tfgan.losses.modified_generator_loss(gan_model)

        # Combine these losses - you can specify more parameters
        # Exactly one of weight_factor and gradient_ratio must be non-None

        combined_loss = non_adversarial_loss_MSE + generator_loss_value
        
    return combined_loss

    
    
    