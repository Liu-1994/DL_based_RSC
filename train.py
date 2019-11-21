# -*- coding: utf-8 -*-
import sys
import keras
import argparse
from tools import *
from models import *
from keras.optimizers import SGD


def run(args):
    data, mod_label, snr_label = load_data_from_hdf5(args.data_path)
    data = np.expand_dims(data, axis=-1)
    np.random.seed(2016)  # 对预处理好的数据进行打包，制作成投入网络训练的格式，并进行one-hot编码
    n_examples = data.shape[0]
    n_train = n_examples * args.train_split  # 对半
    train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
    test_idx = list(set(range(0, n_examples)) - set(train_idx))  # label
    X_train = data[train_idx]
    X_test = data[test_idx]

    Y_train = mod_label[train_idx]
    Y_test = mod_label[test_idx]

    print X_train.shape, X_test.shape

    if args.mode == 'train':
        callbacks = []
        tensorboard = keras.callbacks.TensorBoard(
            log_dir=args.log_path,
            histogram_freq=0, write_graph=True)
        callbacks.append(tensorboard)

        if not os.path.exists(args.checkpoint_path):
            os.makedirs(args.checkpoint_path)

        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.checkpoint_path, 'weights_{epoch:02d}.hdf5'), period=20,
            save_weights_only=False, save_best_only=False)
        callbacks.append(checkpoint)

        num_classes = len(classes)
        model = cnn_model(num_classes)
        # model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['acc'])
        sgd = SGD(lr=args.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])

        history = model.fit(X_train, Y_train, batch_size=args.batch_size, epochs=args.nb_epoch,
                            verbose=2, validation_data=(X_test, Y_test), callbacks=callbacks)

        plot_lines(history.history['acc'], history.history['val_acc'],
                   saved_name=os.path.join(args.log_path, 'acc.png'))
        plot_lines(history.history['loss'], history.history['val_loss'],
                   saved_name=os.path.join(args.log_path, 'loss.png'))

    else:
        model = models.load_model(os.path.join(args.checkpoint_path, 'weights_{:02d}.hdf5'.format(args.nb_epoch)))
        score = model.evaluate(X_test, Y_test, verbose=1, batch_size=args.batch_size)
        print(score)

        acc = {}
        snrs = np.unique(snr_label)
        for snr in snrs:

            # extract classes @ SNR
            test_SNRs = np.where(snr_label[test_idx] == snr)[0]
            test_X_i = X_test[test_SNRs]
            test_Y_i = Y_test[test_SNRs]

            # estimate classes
            test_Y_i_hat = model.predict(test_X_i)
            conf = np.zeros([len(classes), len(classes)])
            confnorm = np.zeros([len(classes), len(classes)])
            for i in range(0, test_X_i.shape[0]):
                j = list(test_Y_i[i, :]).index(1)
                k = int(np.argmax(test_Y_i_hat[i, :]))
                conf[j, k] = conf[j, k] + 1
            for i in range(0, len(classes)):
                confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
            plt.figure()
            plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d dB)" % (snr))
            plt.savefig('{}_dB_confusion_matrix.png'.format(snr), format='png', bbox_inches='tight')
            plt.close()

            cor = np.sum(np.diag(conf))
            ncor = np.sum(conf) - cor
            print ("Overall Accuracy: ", cor / (cor + ncor))
            acc[snr] = 1.0 * cor / (cor + ncor)
        # %%
        # Plot accuracy curve
        plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
        plt.xlabel("Signal to Noise Ratio")
        plt.ylabel("Classification Accuracy")
        plt.title("CNN2 Classification Accuracy on RadioML 2018 dataset")
        plt.savefig('CNN2 Classification Accuracy on RadioML 2018 dataset.png', format='png', bbox_inches='tight')
        plt.close()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/dataset/RadioML/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5')
    parser.add_argument('--log_path', type=str, default='logs')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints')
    parser.add_argument('--nb_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--mode', type=str, default='eval', help="'train' or 'eval")
    parser.add_argument('--train_split', type=float, default=0.5)

    args = parser.parse_args(argv)
    return args

if __name__ == '__main__':
    run(parse_arguments(sys.argv[1:]))

