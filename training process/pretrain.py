import os
import pickle
import imageio
import librosa
import numpy as np
from PIL import Image
import scipy.signal as signal

def Resampling(sig, sampling = 44100, re_sampling = 11000):
    result = int((sig.shape[0]) / sampling * re_sampling)
    x_resampled = signal.resample(sig, result)
    x_resampled = x_resampled.astype(np.float64)
    return x_resampled

# create spectrogram of the Au_data
def Spectrogram(root_path):  
    data = pickle.load(open(root_path+'/Au_data_data.pkl', 'rb'), encoding='bytes')
    Au_data = data['Au_data']

    Au_data0 = Au_data[:, 0]
    Au_data1 = Resampling(Au_data0, sampling = 44100, re_sampling = 11000)
    tmp = librosa.stft(y = Au_data1, n_fft = 510, hop_length = 128, win_length = 510,  window = 'hann', center = True, pad_mode = 'reflect')
    D = librosa.amplitude_to_db(np.abs(tmp), ref = np.max)
    img = np.array(Image.fromarray(D).resize((256, 256)))
    imageio.imwrite(root_path + '/ceptrum0.jpg',img)

    Au_data0 = Au_data[:, 1]
    Au_data1 = Resampling(Au_data0, sampling = 44100, re_sampling = 11000)
    tmp = librosa.stft(y = Au_data1, n_fft = 510, hop_length = 128, win_length = 510,  window = 'hann', center = True, pad_mode = 'reflect')
    D = librosa.amplitude_to_db(np.abs(tmp), ref = np.max)
    img = np.array(Image.fromarray(D).resize((256, 256)))
    imageio.imwrite(root_path + '/ceptrum1.jpg',img)

    Au_data0 = Au_data[:, 2]
    Au_data1 = Resampling(Au_data0, sampling = 44100, re_sampling = 11000)
    tmp = librosa.stft(y = Au_data1, n_fft = 510, hop_length = 128, win_length = 510,  window = 'hann', center = True, pad_mode = 'reflect')
    D = librosa.amplitude_to_db(np.abs(tmp), ref = np.max)
    img = np.array(Image.fromarray(D).resize((256, 256)))
    imageio.imwrite(root_path + '/ceptrum2.jpg',img)

    Au_data0 = Au_data[:, 3]
    Au_data1 = Resampling(Au_data0, sampling = 44100, re_sampling = 11000)
    tmp = librosa.stft(y = Au_data1, n_fft = 510, hop_length = 128, win_length = 510,  window = 'hann', center = True, pad_mode = 'reflect')
    D = librosa.amplitude_to_db(np.abs(tmp), ref = np.max)
    img = np.array(Image.fromarray(D).resize((256, 256)))
    imageio.imwrite(root_path + '/ceptrum3.jpg',img)
    print(root_path,'   comlleted')

if __name__ == "__main__":
    root =os.getcwd()
    classes={'061_foam_brick': 0,
            'green_basketball': 1,
            'salt_cylinder': 2,
            'shiny_toy_gun': 3,
            'stanley_screwdriver': 4,
            'strawberry': 5,
            'toothpaste_box': 6,
            'toy_elephant': 7,
            'whiteboard_spray': 8,
            'yellow_block': 9}
    for key, value in classes.items():
        f = os.walk(root + '/train/' + key + '/' + key)
        for i, j, k in f:
            for num in j:
                Spectrogram(root + '/train/' + key + '/' + key + '/'+num)
            break