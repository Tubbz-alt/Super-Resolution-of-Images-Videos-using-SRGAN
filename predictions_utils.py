

import glob
import cv2
from google.colab.patches import cv2_imshow
import numpy as np
import matplotlib.pyplot as plt


import tensorflow_gan as tfgan
import tensorflow_datasets as tfds
import tensorflow.compat.v1 as tf

tf.compat.v1.enable_resource_variables()


def plot_grid_images(input_array_grid , grid_shape = (5, 4) ,  title = "images_batch" , save_file_path = "./report_images/predictions" , fig_size = (20, 20)) :
  predict_img_grid = tfgan.eval.python_image_grid( input_array_grid , grid_shape=grid_shape)
  plt.figure(figsize=(20,20))
  plt.axis('off')
  plt.title(title)
  plt.imshow(np.squeeze(predict_img_grid ))
  plt.savefig(save_file_path)
  plt.show()
  return 
  
  
def predict_function(numpy_imgs , model_path ):

  predictions_list = []

  with tf.Session(graph=tf.Graph()) as sess:
      meta_graph_def = tf.compat.v1.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_path )
      signature = meta_graph_def.signature_def
      x_tensor_name = signature['serving_default'].inputs['feature'].name
      y_tensor_name = signature['serving_default'].outputs['output'].name
      x = sess.graph.get_tensor_by_name(x_tensor_name)
      y = sess.graph.get_tensor_by_name(y_tensor_name)
      for i in range(len(numpy_imgs)):
        temp_pred = sess.run(y, {x: np.expand_dims(numpy_imgs[i], axis=0 ) } )
        predictions_list.append(np.squeeze(temp_pred , axis=0))
      
      predictions = np.array(predictions_list)

  return predictions



def video_reader(input_file_path):

  """VIDEO READER FUNCTION TO OUTPUT FRAMES AS NUMPY ARRAY . """

  input_capturer = cv2.VideoCapture(input_file_path)

  NUM_OF_FRAMES = int(input_capturer.get(cv2.CAP_PROP_FRAME_COUNT))
  # print(NUM_OF_FRAMES)
  FRAME_WIDTH = int(input_capturer.get(cv2.CAP_PROP_FRAME_WIDTH))
  FRAME_HEIGHT = int(input_capturer.get(cv2.CAP_PROP_FRAME_HEIGHT))

  input_video_buffer = np.empty((NUM_OF_FRAMES, FRAME_WIDTH, FRAME_HEIGHT, 3), np.dtype('uint8'))

  frame_counter = 0 
  ret = True

  while(frame_counter < NUM_OF_FRAMES and ret):
      ret, frame = input_capturer.read()

      # image_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

      # cv2_imshow(frame)

      input_video_buffer[frame_counter] = frame
      frame_counter += 1

      if cv2.waitKey(1) & 0xFF == ord('q') :   ### '27' -> ESCAPE key
          break


  input_capturer.release()
  cv2.destroyAllWindows()

  input_video = input_video_buffer.astype('float')
  # print(input_video.shape)


  return input_video, NUM_OF_FRAMES, FRAME_WIDTH, FRAME_HEIGHT
  

def video_writer(inputs , output_file_path , fps = 6):
  """Array to Video File writer function."""

  FRAME_WIDTH = inputs.shape[2]
  FRAME_HEIGHT = inputs.shape[1]
  ### FOURCC CODEC for compress and decompress strategy

  fourcc_obj = cv2.VideoWriter_fourcc(*'MJPG')

  video_writer = cv2.VideoWriter(output_file_path, fourcc_obj, fps, (FRAME_WIDTH, FRAME_HEIGHT), True)

  for x in range(len(inputs)):
      rescaled_pred_img = inputs[x] * 255.0
      bgr_image = cv2.cvtColor(rescaled_pred_img.astype('uint8') , cv2.COLOR_RGB2BGR)
      video_writer.write(bgr_image)

  video_writer.release()
  cv2.destroyAllWindows()

  return 


### TF 2.0 trials

# imported_model = tf.saved_model.load(export_dir='gpuestimator-tfgan/saved/tmp_export/gan_estimator_export/1584574519')
# imported_model

# imported_model.signatures['serving_default']('input')

# imported_model.signatures['serving_default'].inputs

  
    