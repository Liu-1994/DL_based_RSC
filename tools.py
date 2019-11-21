import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

classes = ['32PSK',
           '16APSK',
           '32QAM',
           'FM',
           'GMSK',
           '32APSK',
           'OQPSK',
           '8ASK',
           'BPSK',
           '8PSK',
           'AM-SSB-SC',
           '4ASK',
           '16PSK',
           '64APSK',
           '128QAM',
           '128APSK',
           'AM-DSB-SC',
           'AM-SSB-WC',
           '64QAM',
           'QPSK',
           '256QAM',
           'AM-DSB-WC',
           'OOK',
           '16QAM']


def load_data_from_hdf5(data_path):
    f = h5py.File(data_path)
    return f['X'].value, f['Y'].value, f['Z'].value


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=classes):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_lines(train_his, val_his, saved_name='images.png'):
    x = np.arange(1, len(train_his)+1)
    plt.plot(x, train_his, color='tomato', linewidth=2, label='train')
    plt.plot(x, val_his, color='limegreen', linewidth=2, label='val')
    plt.legend()
    # plt.show()
    plt.savefig(saved_name, format='png', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    load_data_from_hdf5('/dataset/RadioML/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5')