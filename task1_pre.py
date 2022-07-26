import os
import pickle
import librosa
import imageio
import numpy as np
from PIL import Image
import librosa.display
import scipy.signal as signal

task1_num=248

def Resampling(sig, sampling = 44100, re_sampling = 11000):
    result = int((sig.shape[0]) / sampling * re_sampling)
    x_resampled = signal.resample(sig, result)
    x_resampled = x_resampled.astype(np.float64)
    return x_resampled


# create spectrogram of the Au_data
def Spectrogram(root_path):  
    for n in range(task1_num):
        data = pickle.load(open(root_path+'/Au_data_data.pkl', 'rb'), encoding='bytes')
        Au_data = data['Au_data']

        Au_data0 = Au_data[:, 0]
        Au_data1 = Resampling(Au_data0, sampling = 44100, re_sampling = 11000)
        tmp = librosa.stft(y = Au_data1, n_fft = 510, hop_length = 128, win_length = 510,  window = 'hann', center = True, pad_mode = 'reflect')
        D = librosa.amplitude_to_db(np.abs(tmp), ref = np.max)
        img = np.array(Image.fromarray(D).resize((256, 256)))
        imageio.imwrite(root_path+'/audio_'+'%04d'%n+'0.jpg',img)

        Au_data0 = Au_data[:, 1]
        Au_data1 = Resampling(Au_data0, sampling = 44100, re_sampling = 11000)
        tmp = librosa.stft(y = Au_data1, n_fft = 510, hop_length = 128, win_length = 510,  window = 'hann', center = True, pad_mode = 'reflect')
        D = librosa.amplitude_to_db(np.abs(tmp), ref = np.max)
        img = np.array(Image.fromarray(D).resize((256, 256)))
        imageio.imwrite(root_path+'/audio_'+'%04d'%n+'1.jpg',img)

        Au_data0 = Au_data[:, 2]
        Au_data1 = Resampling(Au_data0, sampling = 44100, re_sampling = 11000)
        tmp = librosa.stft(y = Au_data1, n_fft = 510, hop_length = 128, win_length = 510,  window = 'hann', center = True, pad_mode = 'reflect')
        D = librosa.amplitude_to_db(np.abs(tmp), ref = np.max)
        img = np.array(Image.fromarray(D).resize((256, 256)))
        imageio.imwrite(root_path+'/audio_'+'%04d'%n+'2.jpg',img)

        Au_data0 = Au_data[:, 3]
        Au_data1 = Resampling(Au_data0, sampling = 44100, re_sampling = 11000)
        tmp = librosa.stft(y = Au_data1, n_fft = 510, hop_length = 128, win_length = 510,  window = 'hann', center = True, pad_mode = 'reflect')
        D = librosa.amplitude_to_db(np.abs(tmp), ref = np.max)
        img = np.array(Image.fromarray(D).resize((256, 256)))
        imageio.imwrite(root_path+'/audio_'+'%04d'%n+'3.jpg',img)
        print('%04d'%n,'   comlleted')

def Spectrogram_task(root_path, file):
    data = pickle.load(open(root_path + file, 'rb'), encoding = 'bytes')
    Au_data = data['audio']
    if not os.path.isdir('./task_test/'):
        os.mkdir('./task_test/')
    Au_data0 = Au_data[:, 0]
    Au_data1 = Resampling(Au_data0, sampling = 44100, re_sampling = 11000)
    tmp = librosa.stft(y = Au_data1, n_fft = 510, hop_length = 128, win_length = 510,  window = 'hann', center = True, pad_mode = 'reflect')
    D = librosa.amplitude_to_db(np.abs(tmp), ref = np.max)
    img = np.array(Image.fromarray(D).resize((256, 256)))
    imageio.imwrite('./task_test/' + file.split('.')[0] + '0.jpg', img)
    
    Au_data0 = Au_data[:, 1]
    Au_data1 = Resampling(Au_data0, sampling = 44100, re_sampling = 11000)
    tmp = librosa.stft(y = Au_data1, n_fft = 510, hop_length = 128, win_length = 510,  window = 'hann', center = True, pad_mode = 'reflect')
    D = librosa.amplitude_to_db(np.abs(tmp), ref = np.max)
    img = np.array(Image.fromarray(D).resize((256, 256)))
    imageio.imwrite('./task_test/' + file.split('.')[0] + '1.jpg', img)

    Au_data0 = Au_data[:, 2]
    Au_data1 = Resampling(Au_data0, sampling = 44100, re_sampling = 11000)
    tmp = librosa.stft(y = Au_data1, n_fft = 510, hop_length = 128, win_length = 510,  window = 'hann', center = True, pad_mode = 'reflect')
    D = librosa.amplitude_to_db(np.abs(tmp), ref = np.max)
    img = np.array(Image.fromarray(D).resize((256, 256)))
    imageio.imwrite('./task_test/' + file.split('.')[0] + '2.jpg', img)

    Au_data0 = Au_data[:, 3]
    Au_data1 = Resampling(Au_data0, sampling = 44100, re_sampling = 11000)
    tmp = librosa.stft(y = Au_data1, n_fft = 510, hop_length = 128, win_length = 510,  window = 'hann', center = True, pad_mode = 'reflect')
    D = librosa.amplitude_to_db(np.abs(tmp), ref = np.max)
    img = np.array(Image.fromarray(D).resize((256, 256)))
    imageio.imwrite('./task_test/' + file.split('.')[0] + '3.jpg', img)
    print(file,'   completed')

if __name__ == "__main__":
    root =os.getcwd()
    Spectrogram(root+'/task1/task1/test/')
            