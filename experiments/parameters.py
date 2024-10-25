"""
THIS PYTHON FILE ONLY CONTAINS THE PARAMETERS
FOR THE DIFFUSION MODELS. THIS IS DONE TO AVOID
CIRCULAR IMPORT. 
"""

import logging
import os

""" GENERAL PARAMETERS """
folder_name = "experiments"
img_folder_name = "flow"            # must be in cwd
                                    # where the training dataset is
folder_path = "exp2"                # the folder to create and store all the files 

# Create the folder if it exists
if not os.path.exists(folder_path): 
    os.makedirs(folder_path)

# Configure logging
logging.basicConfig(filename=f'{folder_path}/model_parameters.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


""" TRAINING PARAMETERS """
# TRAINING PARAMETERS
train_model = True                  # controls the "mode" of the ddim.py script
                                    # if True, it's in training mode
                                    # if False, it's in inference mode
load_and_train = False              # if it's true, it loads the model from checkpoint, and trains the model


# MODEL PARAMETERS
image_size = (200, 600)              # for 100x300 is 96, 296 (for 3 downsampling blocks)
                                     # for 200x600 is 200, 600 (for 2 downsampling blocks)

seed = 42
validation_split = 0.2
pad_to_aspect_ratio = False
crop_to_aspect_ratio = True     

num_epochs = 200
batch_size = 12
dataset_repetitions = 1
ema = 0.999
learning_rate = 1e-4
weight_decay = 1e-4

kid_image_size = 75      
plot_diffusion_steps = 20
kid_diffusion_steps = 5

min_signal_rate = 0.02
max_signal_rate = 0.95

embedding_dims = 32
widths = [32, 64, 96, 128] 
block_depth = 2

""" INFERENCE PARAMETERS """
images_to_generate = 5
generate_diffusion_steps = 50

# Log all variables
logging.info(f'GENERAL PARAMETERS:\nfolder_name: {folder_name}\nimg_folder_name: {img_folder_name}\nfolder_path: {folder_path}')
logging.info(f'TRAINING PARAMETERS:\ntrain_model: {train_model}\nload_and_train: {load_and_train}')
logging.info(f'MODEL PARAMETERS:\nimage_size: {image_size}\nseed: {seed}\nvalidation_split: {validation_split}\npad_to_aspect_ratio: {pad_to_aspect_ratio}\ncrop_to_aspect_ratio: {crop_to_aspect_ratio}\nnum_epochs: {num_epochs}\nbatch_size: {batch_size}\ndataset_repetitions: {dataset_repetitions}\nema: {ema}\nlearning_rate: {learning_rate}\nweight_decay: {weight_decay}\nkid_image_size: {kid_image_size}\nplot_diffusion_steps: {plot_diffusion_steps}\nkid_diffusion_steps: {kid_diffusion_steps}\nmin_signal_rate: {min_signal_rate}\nmax_signal_rate: {max_signal_rate}\nembedding_dims: {embedding_dims}\nwidths: {widths}\nblock_depth: {block_depth}')
logging.info(f'INFERENCE PARAMETERS:\nimages_to_generate: {images_to_generate}\ngenerate_diffusion_steps: {generate_diffusion_steps}')










