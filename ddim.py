""" 
THIS DDIM MODEL
TRAINS OUR 100X300 IMAGES

ONLY TENSORFLOW 2.16.1 WILL 
WORK, PLEASE REFER TO THE
OFFICIAL KERAS WEBSITE
https://keras.io/getting_started/
"""

""" ALL IMPORTS """
# import necessary libraries
import warnings
import os
import matplotlib.pyplot as plt 
import tensorflow as tf
import keras
import cv2 as cv
import logging
from datetime import datetime
import shutil

# import from local scripts
from diffusion_model import DiffusionModel
from callbacks import * 
from parameters import *

""" HELPER FUNCTIONS """
def load_dataset(): 
    """
    Loads the dataset for training
    """
    cwd = os.getcwd()
    img_dir = os.path.join(cwd, img_folder_name)
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        img_dir, 
        validation_split = validation_split,
        subset="training", 
        seed = seed,
        image_size = (image_size[0], image_size[1]),  
        batch_size = None,
        shuffle = True,
        crop_to_aspect_ratio = crop_to_aspect_ratio,
        pad_to_aspect_ratio = pad_to_aspect_ratio,
        #labels = labels, 
        #label_mode = None
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        img_dir, 
        validation_split = validation_split,
        subset="validation", 
        seed = seed,
        image_size = (image_size[0], image_size[1]), 
        batch_size = None,
        shuffle = True,
        crop_to_aspect_ratio = crop_to_aspect_ratio,
        pad_to_aspect_ratio = pad_to_aspect_ratio,
        #labels = labels, 
        #label_mode = None
    )
    return train_ds, val_ds

def load_inpainting_data(): 
    """
    Returns the list of files in the image and mask directory for inpainting
    """
    # Get both the directory of the images and the masks
    cwd = os.getcwd()
    inpaint_folder = os.path.join(cwd, inpainting_dir)
    img_folder = os.path.join(inpaint_folder, inpaint_img)
    mask_folder = os.path.joni(inpaint_folder, inpaint_mask)

    # Get the lists of files in both directories
    img_list = os.listdir(img_folder)
    mask_list = os.listdir(mask_folder)

    return sorted(img_list), sorted(mask_list)

def load_inpainting_data_temp(): 
    cwd = os.getcwd()
    mask_folder_name = "mask_and_images"
    inpainting_folder = os.path.join(cwd, mask_folder_name)

    image_name = "image1.jpg"
    mask_name = "image1_mask.jpg"

    image_folder = os.path.join(inpainting_folder, image_name)
    mask_folder = os.path.join(inpainting_folder, mask_name)

    image = cv.imread(image_folder)
    mask = cv.imread(mask_folder)

    return image, mask

def prepare_dataset(train_ds, val_ds): 
    """
    Prepares the dataset for training, used in combination with load_dataset
    """
    train_ds = (train_ds
        .map(normalize_image, num_parallel_calls=tf.data.AUTOTUNE) # each dataset has the structure
        .cache()                                                   # (image, labels) when inputting to 
        .repeat(dataset_repetitions)                               # map
        .shuffle(10 * batch_size)
        .batch(batch_size, drop_remainder=True)
        .prefetch(buffer_size=tf.data.AUTOTUNE))
    val_ds = (val_ds
        .map(normalize_image, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .repeat(dataset_repetitions)
        .shuffle(10 * batch_size)
        .batch(batch_size, drop_remainder=True)
        .prefetch(buffer_size=tf.data.AUTOTUNE)) # THIS IS A PREFETCH DATASET
    return train_ds, val_ds

def normalize_image(images, _):    
    # clip pixel values to the range [0, 1]
    return tf.clip_by_value(images / 255, 0.0, 1.0)

def plot_images(dataset, num_images=5):
    # Create an iterator to get images from the dataset
    iterator = iter(dataset)
    
    # Get the first batch of images
    images = next(iterator)

    # Plot the images
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for i in range(num_images):
        axes[i].imshow(images[i])
        axes[i].axis("off")
    plt.show()

def save_history(history, dict_key_list): 
    for key in dict_key_list: 
        plt.plot(history.history[key], label=key)

    plt.title("Training losses")
    plt.ylabel("Value")
    plt.xlabel("Epochs")

    plt.legend(loc="upper left")
    plt.savefig(f"{folder_path}/training_loss.png")
    plt.close()

""" Main Runtime """
def TrainDiffusionModel():
    """
    THIS SCRIPT IS USED TO TRAIN THE DIFFUSION MODEL
    """

    # Load and prepare the dataset
    train_dataset, val_dataset = load_dataset()
    train_dataset, val_dataset = prepare_dataset(train_dataset, val_dataset)

    # Create and compile the model
    model = DiffusionModel(image_size, widths, block_depth)
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        ),
        loss=keras.losses.mean_absolute_error,
        # Loss function: Pixelwise mean absolute error (MAE).
    )

    # Calculate mean and variance of training dataset for normalization
    model.normalizer.adapt(train_dataset)
        # The adapt method is called on the normalizer using the training dataset.
        # This calculates the mean and variance of the training dataset for normalization.

    if load_and_train: 
        model.load_weights(checkpoint_path)

    """
    Training: 
    - train for at least 50 epochs for good results
    - run training and plot generated images 

    WHEN YOU WANT TO PERFORM INFERENCE BASED ON PREVIOUS
    WEIGHTS, COMMENT THIS BLOCK OUT
    """
    history = model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=val_dataset,
        callbacks=[
            early_stop_callback,
            custom_csv_logger,
            plot_image_callback,
            checkpoint_callback, # checkpoint callback located here
        ],
    )

    #shutil.copy(parameters
    
     #   plt.savefig(f"{folder_path}/training_loss.png")


    # Save the key loss metrics 
    dict_key_list = ["i_loss", "n_loss", "val_i_loss", "val_kid", "val_n_loss"]
    save_history(history, dict_key_list)

def InferenceDiffusionModel(): 
    """
    This script is used for inference, reverse diffusion
    """
    train_dataset, val_dataset = load_dataset()
    train_dataset, val_dataset = prepare_dataset(train_dataset, val_dataset)

    # build and load the model
    model = DiffusionModel(image_size, widths, block_depth)
    model.normalizer.adapt(train_dataset) 
    model.load_weights(checkpoint_path)

    # generate the images
    generated_images = model.generate(images_to_generate, generate_diffusion_steps, True)

    # create a new directory
    generated_dir = os.path.join(folder_path, "generated_images")
    if not os.path.exists(generated_dir): 
        os.makedirs(generated_dir)

    index = 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for image in generated_images: 
        #tf.keras.preprocessing.image.save_img(f"{generated_dir}/generated_img_{index}_at{timestamp}.jpg",image)
        tf.keras.preprocessing.image.save_img(f"{generated_dir}/{timestamp}_generated_img_{index}.jpg", image) 

        index = index + 1

def ContextualInpainting():
    """
    This script is used for inpainting using a trained DDIM
    """
    train_dataset, val_dataset = load_dataset()
    train_dataset, val_dataset = prepare_dataset(train_dataset, val_dataset)

    # build and load the model
    model = DiffusionModel(image_size, widths, block_depth)
    model.normalizer.adapt(train_dataset) 
    model.load_weights(checkpoint_path)

    # load the data
    # THE IMAGES NAME MUST BE THE SAME, OTHERWISE, IT WILL BE CONFUSED WITH
    # img_list, mask_list = load_inpainting_data()
    image, mask = load_inpainting_data_temp()

    # main runtime for each img, and each mask

    print("\n The program is currently running \n")

    inpainted_img = model.inpaint(image, mask, plot_diffusion_steps)

    cv.imshow('Image', inpainted_img) 
    cv.waitKey(0) 
    cv.destroyAllWindows()

    
    # Apply noise to the overall image e.g. modify the sampling process
    # Combine using elementwise multiplication


if __name__ == "__main__":
    """ 
    ENSURE WE ARE USING THE CORRECT VERSION 
    ONLY KERAS 3.6 AND TENSORFLOW 2.16.1 HAS 
    BEEN PROVEN TO WORK WITH THIS SCRIPT
    """
    print(keras.__version__)
    print(tf.__version__)

    print(f"\nNum GPUs Available: {len(tf.config.list_physical_devices('GPU'))}\n")

    warnings.filterwarnings('ignore')
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # disable if JIT Compilation error
    # os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0' 
    # tf.config.optimizer.set_jit(False)

    if runtime == "train":
        try: 
            # initialize logging
            logging.basicConfig(level=logging.INFO, filename=f"{folder_path}/{folder_path}error.log")

            TrainDiffusionModel()
            print(f"\n Finish training for {num_epochs}\n")
        except: 
            logging.error("ERROR: Script stopped running, could be due to memory allocation. Check terminal.")

    elif runtime == "inference":
        if not os.path.exists(folder_path): 
            raise Exception("\nWARNING: Cannot find the directory where the model and all its files are stored\n")
        try: 
            InferenceDiffusionModel()
            print(f"\n Finish generating {images_to_generate}\n")
        except: 
            logging.error("ERROR: Script stopped running. Very Weird. Check terminal.")

    # elif runtime == "inpaint":
    #     if not os.path.exists(folder_path): 
    #         raise Exception("\nWARNING: Cannot find the directory where the model and all its files are stored\n")
    #     try: 
    #         ContextualInpainting()
    #         print(f"\n Finish inpainting all the files in the {inpainting_dir} directory\n")
    #     except: 
    #         logging.error("ERROR: Script stopped running. Very Weird. Check terminal.")









