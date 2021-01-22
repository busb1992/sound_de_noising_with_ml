import numpy as np                                                             
import soundfile as sf
import matplotlib.pyplot as plt
import pywt
import os
from tqdm import tqdm
import math
import itertools

# #############################################################################
# get_all_files
# #############################################################################
path = 'D:/Programing/audio noise reduction with autoencoders/data/files/'                                                  

list_of_files = []
failed = []
samplerate = None
standard_len = None

for curr_file in tqdm(os.listdir(path)):
    curr_data, curr_samplerate = sf.read(path + curr_file)
    if samplerate is None and standard_len is None:
        samplerate = curr_samplerate
        standard_len = len(curr_data)
    if samplerate == curr_samplerate and len(curr_data) >= standard_len:
        list_of_files.append(curr_data[0:standard_len])
    else:
        failed.append(len(curr_data))


'''
plt.plot(data)                                        
dataset = [data, data]                                                      
x_train = np.array(dataset)
'''
# #############################################################################
# wavelet part
# #############################################################################

all_clean_wavelets = []
all_clean_freqs = []
all_noised_wavelet = []
all_noised_freqs = []


scales = np.linspace(1, 110, 100)

orig = 'D:/Programing/audio noise reduction with autoencoders/transformed_data/orig/'
noised = 'D:/Programing/audio noise reduction with autoencoders/transformed_data/noised/'

for curr_idx, curr_data in tqdm(enumerate(list_of_files)):
    coef, freqs = pywt.cwt(data=curr_data, scales=scales, wavelet="morl")
    np.save(orig + str(curr_idx) + '.npy', coef)
    # all_clean_wavelets.append(coef)
    # all_clean_freqs.append(freqs)
    
    noise = np.random.normal(-0.05, 0.05, len(curr_data))
    noised_data = curr_data + noise
    coef, freqs = pywt.cwt(data=noised_data, scales=scales, wavelet="morl")
    np.save(noised + str(curr_idx) + '.npy', coef)
    # all_noised_wavelet.append(coef)
    # all_noised_freqs.append(freqs)

# #############################################################################
# orig wavelet part
# #############################################################################

import keras
from keras import layers
from keras import regularizers

input_img = keras.Input(shape=(100, 93680, 1))

x = layers.Conv2D(32, (3, 3), activation='relu', padding='valid')(input_img)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='valid')(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='valid')(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='valid')(x)
encoded = layers.Dense(32, kernel_initializer='ones',
                              kernel_regularizer=regularizers.l1(0.01),
                              activity_regularizer=regularizers.l2(0.01))(x)


x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='valid')(encoded)
x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='valid')(x)
x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='valid')(x)
# x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='valid')(x)
decoded = layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='valid')(x)

autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.summary()

callbacks_ = [
    keras.callbacks.EarlyStopping(patience=10),
    keras.callbacks.ModelCheckpoint(filepath='model.h5')
    ]



orig_train = 'D:/Programing/audio noise reduction with autoencoders/transformed_data/train/orig/'
noised_train = 'D:/Programing/audio noise reduction with autoencoders/transformed_data/train/noised/'
orig_validate = 'D:/Programing/audio noise reduction with autoencoders/transformed_data/validate/orig/'
noised_validate = 'D:/Programing/audio noise reduction with autoencoders/transformed_data/validate/noised/'


autoencoder.fit_generator(generator(orig_train, noised_train),
                          validation_data=generator(orig_validate, noised_validate),
                          epochs=1000, verbose=2,
                          callbacks=callbacks_, steps_per_epoch=24*4, validation_steps=5*4)


def generator(orig, noised):
    batch_size = 8
    start_points = itertools.cycle(list(range(0, math.floor(len(os.listdir(orig))/batch_size)*batch_size, batch_size)))

    while True:
        curr_start = next(start_points)
        curr_list_orig = []
        curr_list_noised = []
        for curr_orig_item, curr_noised_item in zip(os.listdir(orig)[curr_start:curr_start+batch_size],
                                                    os.listdir(noised)[curr_start:curr_start+batch_size]):
            curr_list_orig.append(np.load(orig + curr_orig_item))
            curr_list_noised.append(np.load(noised + curr_noised_item))
            
        yield (np.asarray(curr_list_noised).reshape((batch_size, 100, 93680, 1)),
               np.asarray(curr_list_orig).reshape((batch_size, 100, 93680, 1)))
    
# #############################################################################
# orig wavelet part
# #############################################################################
 
scales = np.linspace(1, 110, 1000)
#scales = np.array([2 ** x for x in range(1, 7)])
coef, freqs = pywt.cwt(
    data=data, scales=scales, wavelet="morl")
plt.pcolormesh(list(range(len(data))), scales,
                   coef, cmap='Greys')
plt.show()


# reconstruct
mwf = pywt.ContinuousWavelet('morl').wavefun()
y_0 = mwf[0][np.argmin(np.abs(mwf[1]))]

# r_sum = np.transpose(np.sum(np.transpose(coef)/ scales ** 0.5, axis=-1))
# reconstructed = r_sum * (1 / y_0)
r_sum = np.transpose(np.sum(np.transpose(coef)/ scales/32, axis=-1))
reconstructed = r_sum * (1 / y_0)


minused_data = list(np.array(data) - max(data) * 2)
plt.plot(list(range(len(data))), reconstructed, '-k', alpha=0.6, linewidth=10)
plt.plot(list(range(len(data))), data, '-.r', 'x', alpha=0.6)

plt.show()

# #############################################################################
# soundfile write
# #############################################################################

sf.write('new_file.flac', reconstructed, samplerate)

# #############################################################################
# noiseing
# #############################################################################

import numpy as np
noise = np.random.normal(-0.05, 0.05, len(data))
noised_data = data + noise
plt.plot(noised_data)
plt.show()

coef, freqs = pywt.cwt(
    data=noised_data, scales=scales, wavelet="morl")
plt.pcolormesh(list(range(len(data))), scales,
               coef, cmap='Greys')
plt.show()

sf.write('new_file_noised.flac', noised_data, samplerate)
