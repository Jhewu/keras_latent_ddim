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
import numpy as np
import cv2 as cv

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
            #print(f"\nthis is noise_rate {noise_rates}\n")
            #print(f"\nthis is signal_rate {signal_rates}\n")
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
    
    def reverse_diffusion_single(self, initial_noise, diffusion_steps):
        """
        Custom Reverse Diffusion 
        - Performs reverse diffusion (sampling), not simultaneously, but per image
        - The processing time will be longer, but it requires less GPU ram
        """

        step_size = 1.0 / diffusion_steps
        pred_images = []

        for i in range(initial_noise.shape[0]):  # Iterate over each image
            next_noisy_image = initial_noise[i]
            
            for step in range(diffusion_steps):
                noisy_image = next_noisy_image

                # Separate the current noisy image to its components
                diffusion_time = ops.ones((1, 1, 1, 1)) - step * step_size
                noise_rate, signal_rate = self.diffusion_schedule(diffusion_time)          
                pred_noise, pred_image = self.denoise(
                    noisy_image[None, ...], noise_rate, signal_rate, training=False
                ) # Network used in eval mode

                # Remix the predicted components using the next signal and noise rates
                next_diffusion_time = diffusion_time - step_size
                next_noise_rate, next_signal_rate = self.diffusion_schedule(next_diffusion_time)
                next_noisy_image = (
                    next_signal_rate * pred_image + next_noise_rate * pred_noise
                )[0]
                # This new noisy image will be used in the next step
            pred_images.append(pred_image[0])
        return np.stack(pred_images)

    def generate(self, num_images, diffusion_steps, single):
        """
        IMPORTANT
        """
        # noise to images to denormalized images
        # uses reverse diffusion (above)
        initial_noise = keras.random.normal(
            shape=(num_images, image_size[0], image_size[1], 3)
        )
        if single == True:
            generated_images = self.reverse_diffusion_single(initial_noise, diffusion_steps)
            generated_images = self.denormalize(generated_images)
        else:
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
            num_images=batch_size, diffusion_steps=kid_diffusion_steps, single=False
        )
        self.kid.update_state(images, generated_images)

        return {m.name: m.result() for m in self.metrics}

    def plot_images(self, epoch=None, logs=None, num_rows=3, num_cols=3):
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

    def simple_inpaint_1(self, img, mask, diffusion_steps=plot_diffusion_steps):
        """
        The Simplest form of inpainting
        (no contextual awareness)
        """

        # generate the image
        # the generated images are <float32> and 0-1
        initial_noise = keras.random.normal(
            shape=(1, image_size[0], image_size[1], 3)
        )
        generated_images = self.reverse_diffusion_single(initial_noise, diffusion_steps)
        generated_images = self.denormalize(generated_images)

        # create mask and inverted mask
        # normalize them and cast it to a tensor of 32
        norm_mask = mask / 255.0
        norm_inverted_mask = (255 - mask)/255.0
        norm_img = img/255.0

        tf_mask = tf.cast(norm_mask, dtype=tf.float32)
        tf_inverted_mask = tf.cast(norm_inverted_mask, dtype=tf.float32)
        tf_img = tf.cast(norm_img, dtype=tf.float32)

        # print(mask.dtype)
        # print(inverted_mask.dtype)

        # mask it to the image
        generated_image = generated_images[0]
        masked_noise = tf_inverted_mask * generated_image
        masked_image = tf_mask * tf_img
        inpainted_img = masked_noise + masked_image

        return inpainted_img

        # print(generated_images.dtype)
        # print(np.min(generated_image))
        # print(np.max(generated_image))

    def simple_inpaint_2(self, img, mask, diffusion_steps): 
        """
        This one standarizes the image before inpainting
        """
        # create mask and inverted mask
        # normalize them and cast it to a tensor of 32
        norm_mask = mask / 255.0
        norm_inverted_mask = (255 - mask)/255.0
        norm_img = img/255.0

        tf_mask = tf.cast(norm_mask, dtype=tf.float32)
        tf_inverted_mask = tf.cast(norm_inverted_mask, dtype=tf.float32)
        tf_img = tf.cast(norm_img, dtype=tf.float32)

        # generate noise and combine it with inverted mask
        initial_noise = keras.random.normal(shape=(image_size[0], image_size[1], 3))
        masked_noise = tf_inverted_mask * initial_noise

        print(initial_noise.dtype)
        print(tf.math.reduce_max(initial_noise))
        print(tf.math.reduce_min(initial_noise))

        # create image mask and standarize it
        image_standardized = (tf_img - tf.math.reduce_mean(tf_img)) / tf.math.reduce_std(tf_img)
        # print(image_standardized.dtype)
        # print(tf.math.reduce_max(image_standardized))
        # print(tf.math.reduce_min(image_standardized))

        masked_img = image_standardized * tf_mask

        # combine the two images together
        combined_img = masked_img + masked_noise

        # generate the image
        # the generated images are <float32> and 0-1
        expand_img = tf.expand_dims(combined_img, 0)
        generated_images = self.reverse_diffusion_single(expand_img, diffusion_steps)
        generated_images = self.denormalize(generated_images)

        generated_image = generated_images[0]

        plt.imshow(generated_image)
        plt.show()

        return generated_image
    
    def simple_inpaint_3(self, img, mask, diffusion_steps): 
        """
        This one does not standarize the image, thus ending up
        with a white background
        """
        # create mask and inverted mask
        # normalize them and cast it to a tensor of 32
        norm_mask = mask / 255.0
        norm_inverted_mask = (255 - mask)/255.0
        norm_img = img/255.0

        tf_mask = tf.cast(norm_mask, dtype=tf.float32)
        tf_inverted_mask = tf.cast(norm_inverted_mask, dtype=tf.float32)
        tf_img = tf.cast(norm_img, dtype=tf.float32)

        # generate noise and combine it with inverted mask
        initial_noise = keras.random.normal(shape=(image_size[0], image_size[1], 3))
        masked_noise = tf_inverted_mask * initial_noise

        print(initial_noise.dtype)
        print(tf.math.reduce_max(initial_noise))
        print(tf.math.reduce_min(initial_noise))

        # create image mask and standarize it
        # image_standardized = (tf_img - tf.math.reduce_mean(tf_img)) / tf.math.reduce_std(tf_img)
        # print(image_standardized.dtype)
        # print(tf.math.reduce_max(image_standardized))
        # print(tf.math.reduce_min(image_standardized))

        masked_img = tf_img * tf_mask

        # combine the two images together
        combined_img = masked_img + masked_noise

        # generate the image
        # the generated images are <float32> and 0-1
        expand_img = tf.expand_dims(combined_img, 0)
        generated_images = self.reverse_diffusion_single(expand_img, diffusion_steps)
        generated_images = self.denormalize(generated_images)

        generated_image = generated_images[0]

        plt.imshow(generated_image)
        plt.show()

        return generated_image


    def inpaint(self, img, mask, diffusion_steps):
        """
        Simple inpaint (some contextual awareness)

        MIGHT NEED TO CONSIDER THE RANGE OF THE KERAS 
        RANDOM NORMAL

        TRY IT WITHOUT NORMALIZING THE VALUE TO THE 
        NORMAL RANGE


        # self.normalization()

        """
        # create mask and inverted mask
        # normalize them and cast it to a tensor of 32
        norm_mask = mask / 255.0
        norm_inverted_mask = (255 - mask)/255.0
        norm_img = img/255.0

        tf_mask = tf.cast(norm_mask, dtype=tf.float32)
        tf_inverted_mask = tf.cast(norm_inverted_mask, dtype=tf.float32)
        tf_img = tf.cast(norm_img, dtype=tf.float32)

        # generate noise and combine it with inverted mask
        initial_noise = keras.random.normal(shape=(image_size[0], image_size[1], 3))
        masked_noise = tf_inverted_mask * initial_noise

        print(initial_noise.dtype)
        print(tf.math.reduce_max(initial_noise))
        print(tf.math.reduce_min(initial_noise))

        # create image mask and standarize it
        image_standardized = (tf_img - tf.math.reduce_mean(tf_img)) / tf.math.reduce_std(tf_img)
        # print(image_standardized.dtype)
        # print(tf.math.reduce_max(image_standardized))
        # print(tf.math.reduce_min(image_standardized))

        masked_img = image_standardized * tf_mask

        # combine the two images together
        combined_img = masked_img + masked_noise

        # generate the image
        # the generated images are <float32> and 0-1
        expand_img = tf.expand_dims(combined_img, 0)
        generated_images = self.reverse_diffusion_single(expand_img, diffusion_steps)
        generated_images = self.denormalize(generated_images)

        generated_image = generated_images[0]

        # create the mask of the noise
        masked_generated = generated_image * tf_inverted_mask

        masked_img = tf_img * tf_mask

        new_combined_img = masked_img + masked_generated

        plt.imshow(new_combined_img)
        plt.show()

        return new_combined_img

        # print(mask.dtype)
        # print(inverted_mask.dtype)

        # mask it to the image
        # generated_image = generated_images[0]
        # masked_noise = tf_inverted_mask * generated_image
        # masked_image = tf_mask * tf_img
        # inpainted_img = masked_noise + masked_image

        # return inpainted_img

        # print(generated_images.dtype)
        # print(np.min(generated_image))
        # print(np.max(generated_image))





        # # creates the mask and the inverted mask
        # mask = mask                 
        # inverted_mask = 255 - mask 

        # # normalize the masks and the image
        # norm_mask = (mask / 255.0).astype("float32")
        # norm_inverted_mask = (inverted_mask / 255.0).astype("float32")
        # img = (img / 255.0).astype("float32")

        # # multiply by noise, (dtype = float32)
        # initial_noise = np.random.uniform(0, 1, size=(image_size[0], image_size[1], 3))
        # initial_noise = keras.random.normal(shape=(image_size[0], image_size[1], 3))
        # print(initial_noise.dtype)

        # masked_noise = norm_mask * initial_noise

        # # multiply by image 
        # masked_img = norm_inverted_mask * img

        # # combine the image together
        # combined_img = tf.cast(masked_img + masked_noise, dtype=tf.float32)

        # # expand dimension before performing reverse diffusion
        # combined_img = np.expand_dims(combined_img, axis=0)
        
        # generated_images = model.reverse_diffusion_single(combined_img, plot_diffusion_steps)
        # generated_images = model.denormalize(generated_images)

        # print(generated_images[0].dtype)

        # plt.imshow(generated_images[0])
        # plt.show()

    # def inpaint(self, img, mask, diffusion_steps):  # ADD REFINEMENT STEPS (aka "JUMPS")
    #     """
    #     Custom implementation of RePaint by using
    #     the diffusion model's reverse_diffusion with DDIM sampling
    #     """
        
    #     # Normalize the original image and the mask to have a standard deviation of 1, like the noise
    #     image = self.normalizer(img, training=False)
    #     mask = self.normalizer(mask, training=False)

    #     # Step 1: Initialize the random noise
    #     initial_noise = tf.random.normal(shape=(1, image.shape[0], image.shape[1], 3))

    #     print(initial_noise)

    #     # Step 2: Initialize noisy image (start with random noise)
    #     noisy_image = initial_noise

    #     # Step 3: Loop through diffusion steps
    #     step_size = 1.0 / diffusion_steps

    #     print("\n The program is currently running \n")

    #     for step in range(diffusion_steps):
    #         """
    #         Refinement steps ("jumps") can be added later for further corrections
    #         """
    #         # Step 4: Compute current diffusion time (1 - step * step_size)
    #         diffusion_time = tf.ones((1, 1, 1, 1)) - step * step_size

    #         print(diffusion_time)

    #         # Step 5: Compute noise and signal rates based on the diffusion schedule
    #         noise_rate, signal_rate = self.diffusion_schedule(diffusion_time)

    #         print(noise_rate)

    #         # Step 6: Predict noise and reconstructed image using the denoising model
    #         pred_noise, pred_image = self.denoise(
    #             noisy_image[None, ...], noise_rate, signal_rate, training=False
    #         )  # Network in eval mode

    #         print("\n This still works\n")

    #         # **Step 5: Compute Known Pixels**
    #         # The known part of the image should not be updated since we already know it
    #         # Get the known region from the original image
    #         known_image = signal_rate * image + noise_rate * pred_noise

    #         # **Step 6: Compute Unknown Pixels Using DDIM**
    #         # This is deterministic in DDIM, so no need for sampling
    #         # Use the mask to handle the unknown part
    #         unknown_image = signal_rate * pred_image + noise_rate * pred_noise

    #         # **Step 8: Combine known and unknown parts using the mask**
    #         next_image = mask * known_image + (1 - mask) * unknown_image

    #         # Update the noisy image with the combined known and unknown parts
    #         noisy_image = next_image  # Use next_image here for the next iteration!

    #     # Final Step: Return the predicted inpainted image (final denoised image)
    #     return pred_image[0]


    # def inpaint(self, img, mask, diffusion_steps): # ADD REFINEMENT STEPS (aka "JUMPS")
    #     """
    #     Custom inplementation of RePaint by using
    #     the diffusion model's reverse_diffusion
    #     """
        
    #     # normalize the original image and the mask
    #     # to have standard deviation of 1, like the noise
    #     image = self.normalizer(img, training=False)
    #     mask = self.normalizer(mask, training=False)



    #     # Step 1: Initialize the random noise
    #     initial_noise = keras.random.normal(
    #         shape=(image_size[0], image_size[1], 3)
    #     )

    #     # Step 2: Loop through diffusion steps
    #     step_size = 1.0 / diffusion_steps
    #     for step in range(diffusion_steps):
    #         """
    #         Loop through refinements steps here ("jumps")
    #         Will work on this later, after the regular inpainting works
    #         """
    #         noisy_image = next_noisy_image

    #         """MIGHT NEED TO ADD THE IF CONDITION FROM THE BEGINNING"""


    #         # Step 5: Compute Known part
    #         # Calculate what the known (non-masked) pixels of the image should look like at the next time step
    #         diffusion_time = ops.ones((1, 1, 1, 1)) - step * step_size
    #         noise_rate, signal_rate = self.diffusion_schedule(diffusion_time)  


    #         # Step 6: Sample z from unknown part
    #         # This step involves sampling new noise zz specifically for the unknown (masked) regions of the image.



    #         pred_noise, pred_image = self.denoise(
    #             noisy_image[None, ...], noise_rate, signal_rate, training=False
    #         ) # Network used in eval mode





    #         # Remix the predicted components using the next signal and noise rates
    #         next_diffusion_time = diffusion_time - step_size
    #         next_noise_rate, next_signal_rate = self.diffusion_schedule(next_diffusion_time)
    #         next_noisy_image = (
    #             next_signal_rate * pred_image + next_noise_rate * pred_noise
    #         )[0]
    #         # This new noisy image will be used in the next step
    #     return pred_image[0]


    """
        
    function RePaint_DDIM(x0, mask, T, U, model):
    // x0: original image
    // mask: binary mask indicating known (1) and unknown (0) regions
    // T: number of total diffusion steps
    // U: number of refinement steps
    // model: trained diffusion model (εθ)

    // Step 1: Initialize random noise
    xT ~ N(0, I) // Generate random noise

    // Step 2: Loop through diffusion steps
    for t from T down to 1 do
        // Step 3: Loop through refinement steps
        for u from 1 to U do
            // Step 4: Sample noise
            if t > 1 then
                ϵ ~ N(0, I) // Sample noise
            else
                ϵ = 0 // No noise at the final step

            // Step 5: Compute known part
            x_known_t_minus_1 = sqrt(ᾱ_t) * x0 + (1 - ᾱ_t) * ϵ

            // Step 6: Sample z for unknown part
            if t > 1 then
                z ~ N(0, I) // Sample noise
            else
                z = 0 // No noise at the final step

            // Step 7: Compute unknown part
            x_unknown_t_minus_1 = (1 / sqrt(α_t)) * (xt - β_t * sqrt(1 - ᾱ_t) * εθ(xt, t)) + σ_t * z

            // Step 8: Combine known and unknown parts
            xt_minus_1 = m * x_known_t_minus_1 + (1 - m) * x_unknown_t_minus_1

            // Step 9: Resample xt if not the last refinement step
            if u < U and t > 1 then
                xt ~ N(sqrt(1 - β_t_minus_1) * xt_minus_1, β_t_minus_1 * I)
            end if
        end for
    end for

    // Step 14: Return the final inpainted image
    return x0

        
        
        
        
        
        
        
        
        """

    
