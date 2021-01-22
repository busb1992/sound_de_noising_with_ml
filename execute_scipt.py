import numpy as np
from data_preprocess import DataConverter
from generator import GeneratorAE
import keras
from keras import layers
from keras import regularizers

'''
do not forget to set working directory & have the basic folders
'''

# Data transform
wavelet_scales = scales=np.linspace(1, 110, 100)
batch_size_ = 2

data_conv = DataConverter(wavelet_scales)
data_conv.load_audio_file_calc_wavelet_and_noised_wavelet()

# get_generators
from generator import GeneratorAE
huhu = GeneratorAE(batch_size=batch_size_)
test_gen, num_of_test_epochs, valid_gen, num_of_valid_epochs, train_gen, num_of_train_epochs = huhu.get_generators()

# modell learning
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
decoded = layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='valid')(x)

autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.summary()

callbacks_ = [
    keras.callbacks.EarlyStopping(patience=10),
    keras.callbacks.ModelCheckpoint(filepath='model.h5')
    ]

autoencoder.fit_generator(train_gen,
                          validation_data=valid_gen,
                          epochs=1000, verbose=2,
                          callbacks=callbacks_, steps_per_epoch=num_of_train_epochs,
                          validation_steps=num_of_valid_epochs)