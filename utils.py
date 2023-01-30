from natsort import natsorted
import unicodedata
import os
from tqdm import tqdm
import glob
import soundfile as sf
from sound_util import load_wav
import numpy as np
import pandas as pd

# Sound processing
import librosa

import sklearn
from sklearn.preprocessing import RobustScaler  # StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
from multiprocessing import Process, Manager, Pool, TimeoutError
import parmap


def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(
        x, axis=axis
    )  # sk.minmax_scale() : 최대 최소를 0 ~ 1 로 맞춰준다.


def get_timestamp(df, word):
    """ """
    word = unicodedata.normalize("NFC", word)
    word = word.replace("음성", "voice")
    _df = df.query("filename==@word")

    if len(_df) == 0:
        return list([])

    if len(_df) > 1:
        print(_df)
        print(len(_df))

    return (
        _df.gratitude_start.values[0],
        _df.gratitude_end.values[0],
        _df.anger_start.values[0],
        _df.anger_end.values[0],
    )


def get_label(df, word):
    """ """
    word = unicodedata.normalize("NFC", word)
    word = word.replace("음성", "voice")
    _df = df.query("filename==@word")

    if len(_df) == 0:
        return list([])

    if len(_df) > 1:
        print(_df)
        print(len(_df))

    return int(_df.label.values[0])


def get_m4a_list(dir_data_m4a):
    """ """
    if os.path.exists(dir_data_m4a):
        wav_list = natsorted(glob.glob(os.path.join(dir_data_m4a, "*음성2.m4a")))
        print(f"Files are loaded: {len(wav_list)}")
        del dir_data_m4a
    else:
        print("This path is not exist")

    return wav_list


def get_wav_list(dir_data_wav):
    """ """
    if os.path.exists(dir_data_wav):
        wav_list = natsorted(glob.glob(os.path.join(dir_data_wav, "*." + "wav")))
        print(f"Files are loaded: {len(wav_list)}")
        del dir_data_wav
    else:
        print("This path is not exist")

    return wav_list


def get_csv_list(dir_data_csv):
    """ """
    if os.path.exists(dir_data_csv):
        csv_list = natsorted(glob.glob(os.path.join(dir_data_csv, "*." + "csv")))
        print(f"Files are loaded: {len(csv_list)}")
        del dir_data_csv
    else:
        print("This path is not exist")

    return csv_list


def get_crop_wav(wav, start, end, sr, btime):
    """ """
    start, end = round((start / btime) * sr), round((end / btime) * sr)

    return np.array(wav[start:end])


def save_wav(wav, sr, name):
    """ """
    sf.write(f"/data/emocog_data/voices_m4a/{name}.wav", wav, sr)

def dim_reduction(X_train, X_test, num_comp=10):
    #Dimension Reduction
    """ """
    print('############################ dim_reduction ############################')
    scaler = RobustScaler()
    scaler.fit(X_train)
    x_train_scaled = scaler.transform(X_train)
    x_test_scaled = scaler.transform(X_test)
    print(np.amax(X_train), np.amax(X_test))
    print(np.amax(x_train_scaled), np.amax(x_test_scaled))

    # Dim. Reduction
    pca = PCA(n_components=num_comp)
    x_train_pcs = pca.fit_transform(x_train_scaled)
    x_test_pcs = pca.transform(x_test_scaled)

    return x_train_pcs, x_test_pcs


def size_matching(features_, pad_mode):
    # Check the length distribution
    size_feature = []

    for f in features_:
        if len(f) > 0:
            size_feature.append(f.shape[1])

    size_to_match = int(np.mean(size_feature) * 2)
    print('size_to_match->',size_to_match)

    #size matching
    for i, f in enumerate(features_):

        if f.shape[1] < size_to_match:
            t = int(np.ceil(size_to_match / f.shape[1]))
            if pad_mode == 'edge':

                f = np.pad(f, ((0, 0), (size_to_match, size_to_match)), 'edge')

            elif pad_mode == 'repeat':

                f = np.repeat(f, t, axis=1)

            elif pad_mode == 'tile':

                f = np.tile(f, t)

            elif pad_mode == 'zero':

                f = np.pad(f, ((0, 0), (size_to_match, size_to_match)), 'constant', constant_values=0)
            else:
                raise

        f = f[:,:size_to_match]
        features_[i] = f

    features_ = np.array(features_)
    features_=features_.reshape(features_.shape[0],-1)

    return features_

class GetFeatures:
    """ """
    def __init__(self, csv_path, m4a_list, features_get, params):

        self.csv_path = csv_path,
        self.m4a_list = m4a_list,
        self.features_get  = features_get,
        self.params = params
        self.df = pd.read_csv(self.csv_path)
        self.get_df()

    def get_df(self):

        self.df = pd.read_csv(self.csv_path)
        self.df["filename"] = self.df["filename"].map(lambda x: unicodedata.normalize("NFC", x))
        self.df["filename"] = self.df["filename"].map(lambda x: x.replace("음성", "voice"))
        # df.to_csv("temp.csv",index=False)
        self.df.drop([200, 298], inplace=True)

    def get_crop_wave(self):
        pass
    def melspectrogram(self):
        pass


def get_wav_label(df, wav_path, params):
    """ """
    wav_file = wav_path.split("/")[-1]
    wav_  = None 
    wavs_ = []
    labels_ = []
    label_ = None

    ts_h = get_timestamp(df, wav_file)
    if len(ts_h) != 4:
        print(wav_file, ts_h)

    grat_start, grat_end, sad_start, sad_end = ts_h
    y, sr = load_wav(wav_path, scale=True, sr_out=params.sr, mono=True, s_end=None)
    # resample
    # y = librosa.resample(y, orig_sr=sr, target_sr=8000)

    grat_wav = get_crop_wav(y, grat_start, grat_end, sr, params.wtime)
    sad_wav = get_crop_wav(y, sad_start, sad_end, sr, params.wtime)

    # save_wav(grat_wav, sr,wav_file.split('.')[-2] + '_thank')
    # save_wav(sad_wav, sr,wav_file.split('.')[-2] + '_anger')

    if params.woption == 'grad':

        wav_ = np.array(grat_wav)
        wavs_.append(wav_)
        labels_.append(get_label(df, wav_file))
        label_ = get_label(df, wav_file)

    elif params.woption == 'sad':

        wav_ = np.array(grat_wav)
        wavs_.append(wav_)
        label_ = get_label(df, wav_file)

    elif params.woption == 'grad_sad':

        wav_ = np.concatenate((grat_wav, sad_wav), axis=0)
        wavs_.append(np.array(wav_))
        label_ = get_label(df, wav_file)
        
    else:
        raise 

    return wav_ ,label_, sr

def df_preprocess(csv_path):
    """ """
    df = pd.read_csv(csv_path)
    df["filename"] = df["filename"].map(lambda x: unicodedata.normalize("NFC", x))
    df["filename"] = df["filename"].map(lambda x: x.replace("음성", "voice"))
    # df.to_csv("temp.csv",index=False)
    df.drop([200, 298], inplace=True)
    
    return df

def get_melspec(melspectrogram_dics, labels_dict, wav_path, df, params, idx, m4a_list):

    wav_, label_, sr = get_wav_label(df, wav_path, params)
    y_feat = librosa.feature.melspectrogram(
                    y=wav_,
                    sr=sr,
                    n_fft=params.n_ffts,
                    hop_length=params.n_step,
                    n_mels=params.n_mels,
                )
    y_feat = librosa.power_to_db(y_feat, ref=np.max)
    if len(melspectrogram_dics) % 30 ==0 and len(melspectrogram_dics)!=0: print(f'{round(len(melspectrogram_dics)/len(m4a_list),2)*100}%')
    melspectrogram_dics[idx] = y_feat
    labels_dict[idx] = label_

def get_features_parall(csv_path, m4a_list, features_get, params):

    features = {}
    mfcc_feats = []
    stft_feats = []

    df = df_preprocess(csv_path)
    with Pool(processes=8) as pool:

        m = Manager()
        melspectrogram_dics = m.dict()
        labels_dict = m.dict()
        feat_list, label_list, idx_list = m.list(), m.list(), m.list()

        pool.starmap(get_melspec,[(melspectrogram_dics, labels_dict, wave_path, df, params, idx, m4a_list) for idx, wave_path in enumerate(m4a_list[:])])

    pool.close()
    pool.join()

    melspectrogram_dics = dict(sorted(melspectrogram_dics.items()))
    labels_dict = dict(sorted(labels_dict.items()))
    features_ = list(melspectrogram_dics.values())
    labels_ =  list(labels_dict.values())
    features_ = size_matching(features_, params.pad_mode)
    return  features_, labels_
        

def print_scores(y_test, y_pred):

    print('### svm result ###')
    print(confusion_matrix(y_test, y_pred))
    print('\nAccuracy: {:.4f}\n'.format(accuracy_score(y_test, y_pred)))

    print('Micro Precision: {:.4f}'.format(precision_score(y_test, y_pred, average='micro')))
    print('Micro Recall: {:.4f}'.format(recall_score(y_test, y_pred, average='micro')))
    print('Micro F1-score: {:.4f}\n'.format(f1_score(y_test, y_pred, average='micro')))

    print('Macro Precision: {:.4f}'.format(precision_score(y_test, y_pred, average='macro')))
    print('Macro Recall: {:.4f}'.format(recall_score(y_test, y_pred, average='macro')))
    print('Macro F1-score: {:.4f}'.format(f1_score(y_test, y_pred, average='macro')))
    print('Macro F2-score: {:.4f}'.format(fbeta_score(y_test, y_pred, average='macro', beta=2.0)))
