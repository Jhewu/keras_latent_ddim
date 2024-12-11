"""
THIS SCRIPT CONTAINS OUR EVALUATION 
METRIC FOR THE DIFFUSION MODEL. WE 
ARE USING KID
"""

""" ALL IMPORTS """
# import necessary libraries
import keras
from keras import layers
from keras import ops

# import from local script
from parameters import image_size, kid_image_size

"""
---KID Parameters---
KID = Kernel Inception Distance

Why KID? 
- KID is more suitable for small datasets 
- computationally lighter
- numerically more stable
- simpler to implement
"""

"""
HIGH LEVEL SUMMARY

    - Extract features from real and generated images using a pretrained InceptionV3 model.
    - Compute polynomial kernels between these features.
    - Estimate the squared maximum mean discrepancy (MMD) using average kernel values.
    - Update the KID estimate based on real and generated features2.
"""
@keras.saving.register_keras_serializable()
class KID(keras.metrics.Metric): # inherits from Metrics class
    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

        self.kid_tracker = keras.metrics.Mean(name="kid_tracker")
            # mean metric to track the average KID across batches

        # a pretrained InceptionV3 is used without its classification layer
        # transform the pixel values to the 0-255 range, then use the same
        # preprocessing as during pretraining
        self.encoder = keras.Sequential(
            [
                ###CHANGED IMAGE SIZE HERE### 
                keras.Input(shape=(image_size[0], image_size[1], 3)), 
                layers.Rescaling(255.0), # the input images rescaled to the [0, 255] range, 
                layers.Resizing(height=kid_image_size, width=kid_image_size),    # resized to kid_image_size
                layers.Lambda(keras.applications.inception_v3.preprocess_input), # preprocessed as during InceptionV3
                keras.applications.InceptionV3(                                  # pretraining.
                    include_top=False,
                    input_shape=(kid_image_size, kid_image_size, 3),
                    weights="imagenet",
                ),
                layers.GlobalAveragePooling2D(),
            ],
            name="inception_encoder",
        )
            # InceptionV3 is a (CNN) architecture 
            # It is specifically designed for image classification tasks
            # it addressed overfitting by using parallel layers (Inception modules) with different filter sizes.

    def polynomial_kernel(self, features_1, features_2):
        feature_dimensions = ops.cast(ops.shape(features_1)[1], dtype="float32")
        return (
            features_1 @ ops.transpose(features_2) / feature_dimensions + 1.0
        ) ** 3.0
            # Computes a polynomial kernel between two sets of features.

    def update_state(self, real_images, generated_images, sample_weight=None):
        real_features = self.encoder(real_images, training=False)
        generated_features = self.encoder(generated_images, training=False)

        # compute polynomial kernels using the two sets of features
        kernel_real = self.polynomial_kernel(real_features, real_features)
        kernel_generated = self.polynomial_kernel(
            generated_features, generated_features
        )
        kernel_cross = self.polynomial_kernel(real_features, generated_features)

        # estimate the squared maximum mean discrepancy using the average kernel values
        batch_size = real_features.shape[0]
        batch_size_f = ops.cast(batch_size, dtype="float32")
        
        mean_kernel_real = ops.sum(kernel_real * (1.0 - ops.eye(batch_size))) / (
            batch_size_f * (batch_size_f - 1.0)
        )
        
        mean_kernel_generated = ops.sum(
            kernel_generated * (1.0 - ops.eye(batch_size))
        ) / (batch_size_f * (batch_size_f - 1.0))
        
        mean_kernel_cross = ops.mean(kernel_cross)
        
        kid = mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross
        
        self.kid_tracker.update_state(kid)
            # updates the KID estimate based on real and generated features.

    def result(self):
        return self.kid_tracker.result()
            # returns the current KID estimate.

    def reset_state(self):
        self.kid_tracker.reset_state()
            # resets the KID tracker.