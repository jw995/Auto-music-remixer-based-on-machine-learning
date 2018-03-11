
# coding: utf-8

# In[2]:


import librosa
import os
# Change `gpu0` to `cpu` to run on CPU
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
import numpy as np
import scipy
import theano
import theano.tensor as T
import lasagne
from lasagne.utils import floatX
from IPython.display import Audio, display
from lasagne.layers import InputLayer, Conv1DLayer as ConvLayer
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


get_ipython().system('ls inputs')


# ### Load style and content

# In[ ]:


CONTENT_FILENAME = "inputs/imperial.mp3"
STYLE_FILENAME = "inputs/usa.mp3"


# In[ ]:


display(Audio(CONTENT_FILENAME))
display(Audio(STYLE_FILENAME))


# In[ ]:


# Reads wav file and produces spectrum
# Fourier phases are ignored
N_FFT = 2048
def read_audio_spectum(filename):
    x, fs = librosa.load(filename)
    S = librosa.stft(x, N_FFT)
    p = np.angle(S)
    return np.log1p(np.abs(S[np.newaxis,:,:430])), fs


# In[ ]:


a_content, fs = read_audio_spectum(CONTENT_FILENAME)
a_style, fs = read_audio_spectum(STYLE_FILENAME)

N_SAMPLES = a_content.shape[2]
N_CHANNELS = a_content.shape[1]
a_style = a_style[:, :N_CHANNELS, :N_SAMPLES]


# ### Visualize spectrograms for content and style tracks

# In[ ]:


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Content')
plt.imshow(a_content[0,:400,:])
plt.subplot(1, 2, 2)
plt.title('Style')
plt.imshow(a_style[0,:400,:])
plt.show()


# ### Define net

# In[ ]:


# During our tests, we discovered that it is essential to use extremely large number of conv filters 
# In this example we use single convolution with 4096 filters

N_FILTERS = 4096
inputs = InputLayer((1, N_CHANNELS, N_SAMPLES))
conv = ConvLayer(inputs, N_FILTERS, 11, W=lasagne.init.GlorotNormal(gain='relu'))

# Implementation of losses and optimization is based on artistic style transfer example in lasagne recipes
# https://github.com/Lasagne/Recipes/blob/master/examples/styletransfer/Art%20Style%20Transfer.ipynb
def gram_matrix(x):
    g = T.tensordot(x, x, axes=([2], [2])) / x.shape[2]
    return g

def style_loss(A, X,):
    G1 = gram_matrix(A)
    G2 = gram_matrix(X) 
    loss = ((G1 - G2)**2).sum()
    return loss

def content_loss(A, X):
    return ((A - X)**2).sum()

t = np.zeros_like(a_content)

content_features = lasagne.layers.get_output(conv, a_content)
style_features = lasagne.layers.get_output(conv, a_style)

generated = T.tensor3()
gen_features = lasagne.layers.get_output(conv, generated)

# set ALPHA=1e-3 for more style, or ALPHA=0 to turn off content entirely
ALPHA = 1e-2
loss = style_loss(style_features, gen_features) +            ALPHA * content_loss(content_features, gen_features)
grad = T.grad(loss, generated)

f_loss = theano.function([generated], loss)
f_grad = theano.function([generated], grad)

def eval_loss(x0):
    x0 = floatX(x0.reshape((1, N_CHANNELS, N_SAMPLES)))
    return f_loss(x0).astype('float64')

def eval_grad(x0):
    x0 = floatX(x0.reshape((1, N_CHANNELS, N_SAMPLES)))
    return np.array(f_grad(x0)).flatten().astype('float64')


# ### Run optimization

# In[ ]:


#initialization with zeros or gaussian noise can be used
#zeros don't work with ALPHA=0
#t = floatX(np.random.randn(1, N_CHANNELS, N_SAMPLES))
t = floatX(np.zeros((1, N_CHANNELS, N_SAMPLES)))

res = scipy.optimize.fmin_l_bfgs_b(eval_loss, t.flatten(), fprime=eval_grad, maxfun=500)
t = res[0].reshape((1, N_CHANNELS, N_SAMPLES))
print (res[1])


# ### Invert spectrogram and save the result

# In[ ]:


a = np.zeros_like(a_content[0])
a[:N_CHANNELS,:] = np.exp(t[0]) - 1

# This code is supposed to do phase reconstruction
p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
for i in range(500):
    S = a * np.exp(1j*p)
    x = librosa.istft(S)
    p = np.angle(librosa.stft(x, N_FFT))

OUTPUT_FILENAME = 'outputs/out.wav'
librosa.output.write_wav(OUTPUT_FILENAME, x, fs)


# In[ ]:


print (OUTPUT_FILENAME)
display(Audio(OUTPUT_FILENAME))


# ### Visualize spectrograms

# In[ ]:


plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.title('Content')
plt.imshow(a_content[0,:400,:])
plt.subplot(1,3,2)
plt.title('Style')
plt.imshow(a_style[0,:400,:])
plt.subplot(1,3,3)
plt.title('Result')
plt.imshow(a[:400,:])
plt.show()

