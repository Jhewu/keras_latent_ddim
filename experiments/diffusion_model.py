"""
THIS CODE IS THE MAIN CLASS BUILDER FOR
OUR DIFFUSION MODEL.
"""

""" ALL IMPORTS """
# import necessary libraries
import matplotlib.pyplot as plt # type: ignore
import tensorflow as tf
import keras
from keras import layers
from keras import ops
import os

# import from local scripts
from u_net import get_network
from kid_metric import KID
from parameters import (max_signal_rate, min_signal_rate, 
                      image_size, batch_size, kid_diffusion_steps,
                      ema, plot_diffusion_steps, folder_path)

import time

@keras.saving.register_keras_serializable()
class DiffusionModel(keras.Model): 
            # inherits from Model
    def __init__(self, image_size, widths, block_depth):
            # parameters we have seen when defining the U-Net
        super().__init__()

        self.normalizer = layers.Normalization()                    # for pixel normalization
        self.network = get_network(image_size, widths, block_depth) # obtaining the U-NET
        self.ema_network = keras.models.clone_model(self.network)   # EMA version of the network

    def compile(self, **kwargs):
        # compile method is overridden to create custom metrics
            # such as noise loss, image loss and KID
        # these metrics will be tracked during training
        super().compile(**kwargs)

        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss") # initializing the metrics
        self.kid = KID(name="kid")                                  # from the KID class

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker, self.kid] # initialize class metrics

    def denormalize(self, images):
        # convert the pixel values back to 0-1 range
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return ops.clip(images, 0.0, 1.0)

    def diffusion_schedule(self, diffusion_times):
            # calculates the noise_rates and the signal_rates
        
        # convert diffusion times to angles
        start_angle = ops.cast(ops.arccos(max_signal_rate), "float32")
        end_angle = ops.cast(ops.arccos(min_signal_rate), "float32")

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        # use angles to calculate signal and noise rates
        signal_rates = ops.cos(diffusion_angles)
        noise_rates = ops.sin(diffusion_angles)
        # note that their squared sum is always: sin^2(x) + cos^2(x) = 1

        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
            # predicts noise components and calculate image components
            # it uses the network (either main or EMA) based on the training mode
        
        # the exponential moving average weights are used at evaluation
        if training:
            network = self.network # referencing the get_network() function during init()
        else:
            network = self.ema_network

        # predict noise component and calculate the image component using it
        pred_noises = network([noisy_images, noise_rates**2], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
            # performs reverse diffusion (sampling)
            # generates images from initial noise by iterating over diffusion steps.
            # at each step, it separates noisy images into components, denoises them,
                # and remixes using next signal and noise rates.

        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        # important line:
            # at the first sampling step, the "noisy image" is pure noise
            # but its signal rate is assumed to be nonzero (min_signal_rate)
        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            # separate the current noisy image to its components
            diffusion_times = ops.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times) # obtain noise_rates and signal_rates
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=False
            )
            # network used in eval mode

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
            # this new noisy image will be used in the next step

        return pred_images

    def generate(self, num_images, diffusion_steps):
        """
        IMPORTANT
        """
        # noise to images to denormalized images
        # uses reverse diffusion (above)
        initial_noise = keras.random.normal(
            shape=(num_images, image_size[0], image_size[1], 3)
        )
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)
        generated_images = self.denormalize(generated_images)
            # denormalize is to take an image and convert it back to [0, 256] RGB
        return generated_images

    def train_step(self, images):
        """ 
        IMPORTANT
        """
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=True) # from init()
        noises = keras.random.normal(shape=(batch_size, image_size[0], image_size[1], 3))
            # generate random noises

        # sample uniform random diffusion times
        diffusion_times = keras.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        
        # calculate noise rates and signal rates based on diffusion times.
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
                # denoise images into predicted noise and image components
            pred_noises, pred_images = self.denoise( # denoise is used for training
                noisy_images, noise_rates, signal_rates, training=True
            )

            # compute noise loss (used for training) and image loss (used as a metric).
            noise_loss = self.loss(noises, pred_noises)  # used for training
            image_loss = self.loss(images, pred_images)  # only used as metric
            """
            What is tf.GradientTape()? 
                Allows you to compute gradients during training for various neural network architectures
                These gradients are crucial for backpropagation
                
                Usage:
                    - You create a tf.GradientTape context.
                    - Inside this context, you perform operations (e.g., forward pass, loss computation) 
                        involving TensorFlow variables (usually tf.Variables).
                    - The tape records these operations.
                    - When you exit the context, you can compute gradients with respect to the recorded 
                        operations using the tape.
            """
        
        # compute gradients and update network weights using the optimizer.
        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))
            # The zip(...) function pairs up corresponding elements from gradients and trainable_weights
                # each pair consists of a gradient and the corresponding trainable weight. 

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        # track the exponential moving averages of weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(ema * ema_weight + (1 - ema) * weight)
                # update exponential moving averages of network weights.

        # KID is not measured during the training phase for computational efficiency
        return {m.name: m.result() for m in self.metrics[:-1]}
            # return metrics (excluding KID) for tracking.

    def test_step(self, images):
        # similar to the training step, but without gradient computation.
        # measures the Kernel Inception Distance (KID) between real and generated images.
        # KID computation is computationally demanding, so kid_diffusion_steps should be small.
       
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=False)
        noises = keras.random.normal(shape=(batch_size, image_size[0], image_size[1], 3))

        # sample uniform random diffusion times
        diffusion_times = keras.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises

        # use the network to separate noisy images to their components
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=False
        )

        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(images, pred_images)

        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        # measure KID between real and generated images
        # this is computationally demanding, kid_diffusion_steps has to be small
        images = self.denormalize(images)
        generated_images = self.generate(
            num_images=batch_size, diffusion_steps=kid_diffusion_steps
        )
        self.kid.update_state(images, generated_images)

        return {m.name: m.result() for m in self.metrics}

    def plot_images(self, epoch=None, logs=None, num_rows=3, num_cols=6):
        # generates random images for visual evaluation.
        # plots a grid of generated images.
        # useful for assessing the quality of generated samples during training.

        # plot random generated images for visual evaluation of generation quality
        generated_images = self.generate(
            num_images=num_rows * num_cols,
            diffusion_steps=plot_diffusion_steps,
        )

            # create a new directory
        plot_dir = os.path.join(folder_path, "plot_checkpoints")
        if not os.path.exists(plot_dir): 
            os.makedirs(plot_dir)

        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                plt.imshow(generated_images[index])
                plt.axis("off")
        plt.tight_layout()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        plt.savefig(f"{plot_dir}/checkpoint_inference{timestamp}.png")
        #plt.show()
        plt.close()