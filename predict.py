"""
Script to run generator model.
"""

#%matplotlib inline

from BuildTrain_generator import channels, img_rows, img_cols, noise_dim
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import os

generator = load_model(os.path.join('models','generator.h5')

def show_images(noise):
    generated_images = generator.predict(noise) 
    plt.figure(figsize=(10,10))

    for i, image in enumerate(generated_images):
        plt.subplot(10, 10, i+1)
        for i, image in enumerate(generated_images):
            plt.subplot(10, 10, i+1)
            if channels == 1:
                plt.imshow(image.reshape((img_rows, img_cols)), cmap='gray')
            else:
                plt.imshow(image.reshape((img_rows, img_cols, channels)))
        plt.axis('off')

    plt.tight_layout()
    plt.show()

noise = np.random.normal(0, 1, size=(100, noise_dim))
show_images(noise)




