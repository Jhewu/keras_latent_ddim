"""
THIS SCRIPT CONTAINS ALL THE CALLBACK FUNCTIONS 
FOR THE DIFFUSION MODEL
"""

""" ALL IMPORTS """
# import necessary libraries
import keras
import csv

#import form local script
from parameters import folder_path

""" 
Create Custom Callback 
This callback only sample and plot
the images after each 10 epochs, 
saving computer resource
"""
class CustomCallback(keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs=None): 
        if (epoch + 1) % 10 == 0: 
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
    monitor="val_kid",
    mode="min",
    save_best_only=True,)

plot_image_callback = CustomCallback()

"""
Early Stopping Callback
Ensure we are not wasting resources
"""
early_stop_callback = keras.callbacks.EarlyStopping(
    monitor="val_kid", 
    min_delta=1e-3,
    patience=3,
    verbose=1,
    mode="min",
    restore_best_weights=True,
    start_from_epoch=50,
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
        self.writer.writerow(['epoch', 'accuracy', 'val_accuracy', 'val_loss'])

    def on_epoch_end(self, epoch, logs=None):
        row = [epoch, logs.get('accuracy'), logs.get('val_accuracy'), logs.get('val_loss')]
        self.writer.writerow(row)

    def on_train_end(self, logs=None):
        self.file.close()

custom_csv_logger = CustomCSVLogger(f"{folder_path}/csv_callback.csv")