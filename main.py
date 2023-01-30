# Python libraries:
import os
import time
import sys
import copy

# Data processing
import csv
import glob
from natsort import natsorted
import numpy as np
import pandas as pd

# pd.set_option("display.max.columns", None)

# Etc
from tqdm.notebook import tqdm
from multiprocessing import Process, Manager, Pool, TimeoutError

# from multiprocessing.pool import ThreadPool, get_context

# Visualization
import matplotlib
import matplotlib.pyplot as plt
import IPython.display as ipd
import librosa.display
import seaborn as sns
import argparse

# Sound processing
import librosa

# Feature processing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler  # StandardScaler, MinMaxScaler

# Training
from sklearn import svm

# Evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
)

# Utils
from utils import get_m4a_list, size_matching, dim_reduction, get_features_parall
from ml import train_svm, train_xgboost
from multiprocessing import Process, Manager, Pool, TimeoutError


def get_parser():

    parser = argparse.ArgumentParser("argument for training")
    parser.add_argument("--n_ffts", type=int, default=2048, help="")
    parser.add_argument("--n_step", type=int, default=512, help="")
    parser.add_argument("--n_wins", type=int, default=1024, help="")
    parser.add_argument("--n_mels", type=int, default=64, help="")
    parser.add_argument("--n_mfcc", type=int, default=20, help="")
    parser.add_argument("--y_scale", type=bool, default=True, help="")
    parser.add_argument("--y_mono", type=bool, default=True, help="")
    parser.add_argument("--sr", type=int, default=22050, help="")
    parser.add_argument("--num_pc", type=int, default=10, help="")
    parser.add_argument("--svm_cost", type=int, default=1, help="")
    parser.add_argument("--svm_type", type=int, default=0, help="")
    parser.add_argument("--svm_kernel", type=int, default=0, help="")
    parser.add_argument(
        "--data_path", type=str, default="", help=""
    )
    parser.add_argument(
        "--csv_path", type=str, default="", help=""
    )
    parser.add_argument("--target_file", type=str, default="", help="")
    parser.add_argument("--feature_name", type=str, default="melspectrogram", help="")
    parser.add_argument("--wtime", type=int, default=1000, help="")
    parser.add_argument("--woption", type=str, default='', help="grad")
    parser.add_argument("--pad_mode", type=str, default='', help="zero")


    opt = parser.parse_args()
    return opt


def main():

    opt = get_parser()
    print(opt)
    m4a_list = get_m4a_list(opt.data_path)
    features, labels = None, None
    start_time = time.time()
    features, labels = get_features_parall(opt.csv_path, m4a_list, opt.feature_name, opt)
    print("--- %s seconds ---" % (time.time() - start_time))
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=2021)
    x_train_pcs, x_test_pcs = dim_reduction(X_train, X_test, 10)
    train_svm(1, x_train_pcs, y_train, x_test_pcs, y_test)
    train_xgboost(X_train,  y_train, X_test, y_test)

if __name__ == "__main__":
    main()
