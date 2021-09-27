from matplotlib import pyplot as plt
from scipy.io import wavfile
import numpy as np
import sys
from python_speech_features import mfcc, logfbank
import speech_recognition as speech_recog

def get_mfcc(archivo, clase):
    freq_sample, sig_audio = wavfile.read(archivo)

    # Output the parameters: Signal Data Type, Sampling Frequency and Duration
    #print('\nShape of Signal:', sig_audio.shape)
    #print('Signal Datatype:', sig_audio.dtype)
    #print('Signal duration:', round(sig_audio.shape[0] / float(freq_sample), 2), 'seconds')

    pow_audio_signal = sig_audio / np.power(2, 15)

    # pow_audio_signal = pow_audio_signal[:100]
    time_axis = 1000 * np.arange(0, len(pow_audio_signal), 1) / float(freq_sample)

    # x = tiempo (ms)
    # y = amplitud
    ### plt.plot(time_axis, pow_audio_signal, color='red')
    ### plt.show()

    # Working on the same input file
    # Extracting the length and the half-length of the signal to input to the foruier transform

    sig_length = len(sig_audio)
    half_length = np.ceil((sig_length + 1) / 2.0).astype(int)

    # We will now be using the Fourier Transform to form the frequency domain of the signal
    signal_freq = np.fft.fft(sig_audio)

    # Normalize the frequency domain and square it
    signal_freq = abs(signal_freq[0:half_length]) / sig_length
    signal_freq **= 2
    transform_len = len(signal_freq)

    # The Fourier transformed signal now needs to be adjusted for both even and odd cases
    if sig_length % 2:
        signal_freq[1:transform_len] *= 2
    else:
        signal_freq[1:transform_len - 1] *= 2

    # Extract the signal's strength in decibels (dB)
    exp_signal = 10 * np.log10(signal_freq)

    x_axis = np.arange(0, half_length, 1) * (freq_sample / sig_length) / 1000.0

    # x = frecuencia (khz)
    # y = poder de señal (dB)
    ### plt.plot(x_axis, exp_signal, color='green', linewidth=1)
    ### plt.show()

    sampling_freq, sig_audio = wavfile.read(archivo)

    # We will now be taking the first 15000 samples from the signal for analysis
    sig_audio = sig_audio[:15000]

    # Using MFCC to extract features from the signal
    mfcc_feat = mfcc(sig_audio, sampling_freq)
    ### print('\nMFCC Parameters\nWindow Count =', mfcc_feat.shape[0])
    ### print('Individual Feature Length =', mfcc_feat.shape[1])

    mfcc_feat = mfcc_feat.T

    # mfcc features
    ### plt.matshow(mfcc_feat)
    ### plt.show()

    #print("\nMFCC")

    #print(mfcc_feat)

    mfcc_data = [clase]
    for i in mfcc_feat:
        mfcc_data.append(np.mean(i))
        #print(clase, np.average(i))

    ### print("Archivo: ", archivo, " - ", mfcc_data)
    return mfcc_data

'''
print("\nMFCC shape (2)")
print(mfcc_feat.shape)

#print("\nBanco de filtros")

fb_feat = logfbank(sig_audio, sampling_freq)
print('\nFilter bank\nWindow Count =', fb_feat.shape[0])
print('Individual Feature Length =', fb_feat.shape[1])

fb_feat = fb_feat.T
plt.matshow(fb_feat)
plt.show()

#np.set_printoptions(threshold=sys.maxsize)
print(fb_feat)
'''

'''
rec = speech_recog.Recognizer()

mic_test = speech_recog.Microphone()

speech_recog.Microphone.list_microphone_names()

with speech_recog.Microphone(device_index=1) as source:
    rec.adjust_for_ambient_noise(source, duration=3)
    print("Reach the Microphone and say something!")
    audio = rec.listen(source)


try:
    print("I think you said: \n" + rec.recognize_google(audio))
except Exception as e:
    print(e)'''