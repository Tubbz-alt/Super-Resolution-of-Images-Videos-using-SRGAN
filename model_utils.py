

import shutil
import os
import glob


def best_model_saver (model_dir ,  best_chkpt_monitoring_path , best_chkpt_monitoring_variable):
    best_model_dir = model_dir + '/best_chkpt'

    try:
        shutil.rmtree(best_model_dir)
    except:
        pass

    os.makedirs(best_model_dir)

    src_dir = model_dir + '/model.ckpt-' + str(best_chkpt_monitoring_path) + ".*"
    print(src_dir)
    dest_dir = best_model_dir
    for file in glob.glob(src_dir):
          print("Following file is copied to best model directory :" + file)
          shutil.copy(file, dest_dir)
    
    try:
      os.remove(model_dir + "/chkpt_file.txt")
    except:
      pass

    info_written = False
    ### Save the best model weights checkpoint step to file 
    with open(model_dir + "/chkpt_file.txt","w+") as best_chkpt_file:
        best_chkpt_file.write("best_step" + " " + str(best_chkpt_monitoring_path)+"\n")
        best_chkpt_file.write("best_psnr_score" + " " + str(best_chkpt_monitoring_variable)+"\n")
        info_written = True
    best_chkpt_file.close()
    
    return info_written
  

def get_best_parameters(chkpt_num):
  src_dir = model_dir + '/best_chkpt' +'/model.ckpt-' + str(chkpt_num) + ".*"
  counter = 0
  dest_dir = best_model_dir
  for file in glob.glob(src_dir):
            counter = counter + 1
            print("Following file is copied to save model directory {} : {} " .format(dest_dir , file))
            shutil.copy(file, dest_dir)

  SAVED_FLAG = counter == 3
  return SAVED_FLAG
  
    