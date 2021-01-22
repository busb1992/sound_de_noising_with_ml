def morlet2_custom(self, M, s, w=5):
        # Python 2.7 can't support cwt with morlet2 wavelet function
        # Therefore we needed to export it from up-to-date version of scipy.
        # If code would be upgraded to python 3.x and new version of scipy
        # could be used, original funtions can be used as well.
        # s = float(s)
        x = np.arange(0, M) - (M - 1.0) / 2
        x = x / s
        wavelet = np.exp(1j * w * x) * np.exp(-0.5 * x**2) * np.pi**(-0.25)
        output = np.sqrt(1/s) * wavelet
        return output

    def cwt_custom(self, data, widths, dtype=None, **kwargs):
        # Python 2.7 can't support cwt with morlet2 wavelet function
        # Therefore we needed to export it from up-to-date version of scipy.
        # If code would be upgraded to python 3.x and new version of scipy
        # could be used, original funtions can be used as well.
        if dtype is None:
            if np.asarray(self.morlet2_custom(1,
                                              widths[0],
                                              **kwargs)).dtype.char in 'FDG':
                dtype = np.complex128
            else:
                dtype = np.float64

        output = np.zeros((len(widths), len(data)), dtype=dtype)
        for ind, width in enumerate(widths):
            N = np.min([10 * width, len(data)])
            wavelet_data = np.conj(self.morlet2_custom(N,
                                                       width,
                                                       **kwargs)[::-1])
            output[ind] = convolve(data, wavelet_data, mode='same')
        return output

def calc_wavelet(self, data):

    wave_matrix = self.cwt_custom(data, self.widths, w=self.wavelet_w)
    # wave_matrix = signal.cwt(data, signal.morlet2, self.widths,
    #                         w=self.wavelet_w)
    return np.round(np.abs(wave_matrix)/np.abs(wave_matrix).max(), 4)

def set_wavelet_defaults_for_calc(self, signal_len_to_set):
    self.time, self.deltatime = np.linspace(0,
                                            signal_len_to_set/self.sampling_rate,
                                            signal_len_to_set,
                                            retstep=True)
    self.fs = 1/self.deltatime
    self.freq = np.linspace(self.wavelet_freq_min, self.wavelet_freq_max,
                            self.wavelet_freqsteps)
    self.widths = self.wavelet_w * self.fs / (2 * self.freq * np.pi)



def plot_wavelet(self, wavelet_2_plot, save_name='not_specified'):

    try:
        plt.ioff()
        wavelet_2_plot = cv2.resize(wavelet_2_plot, None,
                                    fx=float(float(len(self.time_4_plot))/float(len(self.time))),
                                    fy=float(float(len(self.freq_4_plot))/float(len(self.freq))),
                                    interpolation=cv2.INTER_LINEAR)
        plt.pcolormesh(self.time_4_plot, self.freq_4_plot,
                       np.abs(wavelet_2_plot), cmap='viridis')
        if self.add_time_stamp_to_image:
            plt.savefig(self.save_fig + self.curr_date_and_time +
                        '_' + save_name +
                        '.png', dpi=400)
        else:
            plt.savefig(self.save_fig + save_name +
                        '.png', dpi=400)
        plt.clf()
    except AttributeError:
        self.set_wavelet_defaults_for_pic_gen()
        self.plot_wavelet(wavelet_2_plot, save_name)


wavelet_w = 20.
time, deltatime = np.linspace(0,
                              len(data)/samplerate,
                              len(data),
                              retstep=True)
fs = 1/deltatime
freq = np.linspace(1, 5000, 1000)
widths = wavelet_w * fs / (2 * freq * np.pi)

'''
from scipy.signal import cwt
from scipy.signal import morlet2

matrix = cwt(data, morlet2, widths)

plt.pcolormesh(list(range(len(data))), widths,
               matrix2, cmap='viridis')
plt.show()

#####

coef, freqs = pywt.cwt(
    data=data, scales=freq, wavelet="morl"
)
'''
















 # orig
scales = np.array([2 ** x for x in range(70)])
coef, freqs = pywt.cwt(
    data=my_singal, scales=scales, wavelet="morl"
)
I can reconstruct it using the following ...

mwf = pywt.ContinuousWavelet('morl').wavefun()
y_0 = mwf[0][np.argmin(np.abs(mwf[1]))]

r_sum = np.transpose(np.sum(np.transpose(coef)/ scales ** 0.5, axis=-1))
reconstructed = r_sum * (1 / y_0)
