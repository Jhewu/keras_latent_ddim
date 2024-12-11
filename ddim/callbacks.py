"""
THIS SCRIPT CONTAINS ALL THE CALLBACK FUNCTIONS 
FOR THE DIFFUSION MODEL
"""

""" ALL IMPORTS """
# import necessary libraries
import keras
import csv
import tensorflow as tf
import numpy

#import form local script
from parameters import folder_path, checkpoint_monitor, early_stop_monitor, early_stop_patience, early_stop_start_epoch, early_stop_min_delta, plot_on_epoch

""" 
Create Custom Callback 
This callback only sample and plot
the images after each 10 epochs, 
saving computer resource
"""
class CustomCallback(keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs=None): 
        if (epoch + 1) % plot_on_epoch == 0: 
            self.model.plot_images()

"""
Checkpoint Callback
Save the best performing models
only
"""
checkpoint_path = f"{folder_path}/diffusion_model.weights.h5"
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor=checkpoint_monitor,
    mode="min",
    save_best_only=True,)

plot_image_callback = CustomCallback()

"""
Early Stopping Callback
Ensure we are not wasting resources
"""
early_stop_callback = keras.callbacks.EarlyStopping(
    monitor=early_stop_monitor, 
    min_delta=early_stop_min_delta,
    patience=early_stop_patience,
    verbose=1,
    mode="min",
    restore_best_weights=True,
    start_from_epoch=early_stop_start_epoch,
)

"""
CSV Logger Callback
Creates a csv file that logs 
epoch, acc, val_acc, val_loss
"""

class CustomCSVLogger(keras.callbacks.Callback):
    def __init__(self, file_path):
        super(CustomCSVLogger, self).__init__()
        self.file_path = file_path
        self.file = None

    def on_train_begin(self, logs=None):
        self.file = open(self.file_path, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['i_loss', 'n_loss', 'val_ai_loss', 'val_kid', "val_n_loss"])

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None: 
                row = [epoch]
                for metrics in ["i_loss", "n_loss", "val_i_loss", "val_kid", "val_n_loss"]:
                    value = logs.get(metrics, "N/A")
                    if isinstance(value, tf.Tensor): 
                         value = value.numpy()
                    row.append(value) # add "N/A" if the metric is not found
                # row = [epoch, logs.get('accuracy'), logs.get('val_accuracy'), logs.get('val_loss')]
                self.writer.writerow(row)

    def on_train_end(self, logs=None):
        self.file.close()

custom_csv_logger = CustomCSVLogger(f"{folder_path}/csv_callback.csv")
