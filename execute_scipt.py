import numpy as np
from data_preprocess import DataConverter

# do not forget to set working directory & have the basic folders

wavelet_scales = scales=np.linspace(1, 110, 100)

data_conv = DataConverter(wavelet_scales)
data_conv.load_audio_file_calc_wavelet_and_noised_wavelet()