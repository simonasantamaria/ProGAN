import pickle
import numpy as np
import tensorflow as tf
import PIL.Image
from PIL import Image
import numpy as np

# Initialize TensorFlow session.
tf.InteractiveSession()

# Import official CelebA-HQ networks.
#with open('/content/drive/MyDrive/ryver.ai/06_Technology/CGAN/NIH/GAN_NIH/karras2018iclr-celebahq-1024x1024.pkl', 'rb') as file:
with open('/content/drive/MyDrive/ryver.ai/06_Technology/CGAN/NIH/GAN_NIH/network-final.pkl', 'rb') as file:
    G, D, Gs = pickle.load(file)


# Generate latent vectors.
latents = np.random.RandomState(1000).randn(1000, *Gs.input_shapes[0][1:]) # 1000 random latents
latents = latents[[477, 56, 83, 887, 583, 391, 86, 340, 341, 415]] # hand-picked top-10

# Generate dummy labels (not used by the official networks).
labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])

# Run the generator to produce a set of images.
images = Gs.run(latents, labels)

# Convert images to PIL-compatible format.
images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8) # [-1,1] => [0,255]
images = images.transpose(0, 2, 3, 1) # NCHW => NHWC

# Save images as PNG.
print("images.shape[0]",images.shape[0])
for idx in range(images.shape[0]):
    print(idx)
    print("images",images.shape)
    print("i0",images[idx].shape)
    print("ai",type(images))
    PIL.Image.fromarray(images[idx], 'RGB').save('img%d.png' % idx)