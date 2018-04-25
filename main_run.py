import scipy.misc as sc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
import librosa

import vgg16
import style_trans as trans
import read_audio as audio2spec


# check tensorflow version
tf.__version__

# check if model has been downloaded
vgg16.maybe_download()

# to transfer a wav file to spectrogram
CONTENT_FILENAME = "inputs/fg.wav"
STYLE_FILENAME = "inputs/bk.wav"
#display(Audio(CONTENT_FILENAME))
#display(Audio(STYLE_FILENAME))


# read audio
a_content, fs = audio2spec.read_audio_spectum(CONTENT_FILENAME)
a_style, fs = audio2spec.read_audio_spectum(STYLE_FILENAME)
print('sampling frequency',fs)

N_SAMPLES = a_content.shape[1]
N_CHANNELS = a_content.shape[0]
a_style = a_style[:N_CHANNELS, :N_SAMPLES]
print(a_content.shape)
# 430 1025


# preprocess content image 
content_image = a_content
plt.imshow(content_image)

height, width = content_image.shape
nchannels = 3

content=np.zeros((height, width, nchannels));

content_para=content_image.max()
content_image=content_image*1/content_image.max()*255;
content[:,:,0] = content_image;
content[:,:,1] = content_image;
content[:,:,2] = content_image;

plt.imshow(content)
print(content.shape)

content_image=content



# preprocess style image 
style_image = a_style

height, width = style_image.shape
nchannels = 3
print(style_image.shape)

style_image=style_image*1/content_para*255
style_para=style_image.max()

style=np.zeros((height, width, nchannels));
style[:,:,0] = style_image;
style[:,:,1] = style_image;
style[:,:,2] = style_image;

plt.imshow(style)
print(style.shape)

style_image=style



# define content layer and style layer
content_layer_ids = [4]
style_layer_ids = list(range(13))
# can also select a sub-set of the layers,
# e.g. style_layer_ids = [1, 2, 3, 4]


# style-transfer
img = trans.style_transfer(content_image=content_image,
                     style_image=style_image,
                     content_layer_ids=content_layer_ids,
                     style_layer_ids=style_layer_ids,
                     weight_content=1.5,
                     weight_style=10.0,
                     weight_denoise=0.3,
                     num_iterations=60,
                     step_size=10.0)

sc.imsave('bk_and_fg.png', img)
print(img.shape)


# convert the 3-channel back to greyscale
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

img0 = mpimg.imread('bk_and_fg.png')
mixed_gray = rgb2gray(img0)   
mixed_gray = mixed_gray*content_para
plt.imshow(mixed_gray, cmap = plt.get_cmap('gray'))
plt.show()

print(mixed_gray.shape)


# convert the spectrogram back to audio
a=mixed_gray
print('inverting...')
fs=22050

# phase reconstruction
N_FFT = 2048

p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
for i in range(500):
    S = a * np.exp(1j*p)
    x = librosa.istft(S)
    p = np.angle(librosa.stft(x, N_FFT))

OUTPUT_FILENAME = 'bk_and_fg.wav'
librosa.output.write_wav(OUTPUT_FILENAME, x, fs)
print('All done!')

