import os
import pywt
from tqdm import tqdm
import numpy as np
import soundfile as sf


class DataConverter:
    samplerate = None
    standard_len = None
    noise_boundary = 0.05

    def __init__(self, scales=np.linspace(1, 110, 100)):
        self.scales = scales

    def load_audio_file_calc_wavelet_and_noised_wavelet(self,
                                                        audio_files_loc=os.getcwd()+'/data/files/',
                                                        output_files_loc=os.getcwd()+'/transformed_data/'):

        loaded_files = self._load_files_(audio_files_loc)
        noised_files = self._noise_signal_(loaded_files)
        self._transform_and_save_wavelet_(loaded_files, output_files_loc + 'orig/')
        self._transform_and_save_wavelet_(noised_files, output_files_loc + 'noised/')

    def _load_files_(self, path):
        list_of_files = []

        for curr_file in tqdm(os.listdir(path)):
            curr_data, curr_samplerate = sf.read(path + curr_file)
            if self.samplerate is None and self.standard_len is None:
                samplerate = curr_samplerate
                standard_len = len(curr_data)
            if samplerate == curr_samplerate and len(curr_data) >= standard_len:
                list_of_files.append(curr_data[0:standard_len])
        return list_of_files

    def _noise_signal_(self, files_to_noise):
        noised_signals = []
        for curr_singal in tqdm(files_to_noise):
            noise = np.random.normal(self.noise_boundary*-1,
                                     self.noise_boundary, len(curr_singal))
            noised_signals.append(curr_singal + noise)
        return noised_signals

    def _transform_and_save_wavelet_(self, list_of_files, folder_to_save):
        for curr_idx, curr_data in tqdm(enumerate(list_of_files)):
            try:
                matrix = self._calc_wavelet_(curr_data)
                np.save(folder_to_save + str(curr_idx) + '.npy', matrix)
            except Exception as exc:
                print('Error occured: ' + str(exc))

    def _calc_wavelet_(self, signal_to_calculate):
        coef, _ = pywt.cwt(data=signal_to_calculate, scales=self.scales,
                           wavelet="morl")
        return coef

    def reconstruct_wavelet(self, matrix):
        mwf = pywt.ContinuousWavelet('morl').wavefun()
        y_0 = mwf[0][np.argmin(np.abs(mwf[1]))]
        r_sum = np.transpose(np.sum(np.transpose(matrix)/ self.scales/32,
                                    axis=-1))
        reconstructed = r_sum * (1 / y_0)
        return reconstructed
