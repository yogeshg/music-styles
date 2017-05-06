import logging
logging.basicConfig(level = logging.INFO , format=
        '%(asctime)s:%(levelname)s:%(name)s:%(threadName)s:line %(lineno)d: %(message)s')
logger = logging.getLogger(__name__)

import cPickle
import numpy as np
import math
import json
import sys
import pandas as pd
import os
from sklearn.metrics import confusion_matrix

import keras
from keras.layers import Input, Embedding, Conv1D, GlobalMaxPool1D, Dense, GlobalAvgPool1D, Dropout, BatchNormalization
from keras.layers import concatenate
from keras.models import Model, model_from_json
from keras.preprocessing import sequence
from keras import regularizers
from keras.engine.topology import Layer
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

from util import plot_model, plot_metric, save_code, fill_dict
from util.archiver import get_archiver
import config as c

# MAX_CHORDS = 5587
# MAX_LABELS = 5
NUM_NOTES = 88
NUM_DIM = 1024

class LogSumExpPooling(Layer):

    def call(self, x):
        # could be axis 0 or 1
        return tf.reduce_logsumexp(x, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[:1]+input_shape[2:]

def get_conv_stack(input_layer, filters, kernel_sizes, activation, kernel_l2_regularization, dropout_rate):
    layers = [Conv1D(activation=activation, padding='same', strides=1, filters=filters, kernel_size = size,
                kernel_regularizer=regularizers.l2(kernel_l2_regularization))(input_layer) for size in kernel_sizes]
    if (len(layers) <= 0):
        return input_layer
    elif (len(layers) == 1):
        return Dropout(dropout_rate, noise_shape=None, seed=None)(layers[0])
    else:
        return Dropout(dropout_rate, noise_shape=None, seed=None)(concatenate(layers))

def get_model(embeddings=True):
    params = {k:v for k,v in locals().iteritems() if k!='weights'}
    # x = Input(shape=(NUM_NOTES,), dtype='float32')
    x = Input(shape=(MAX_CHORDS,NUM_NOTES), dtype='float32')
    if embeddings:
        y1 = Dense(NUM_DIM, activation='linear', use_bias=False, weights=[M1], trainable=False)(x)
    else:
        y1 = x
    #y2 = BatchNormalization()(y1)
    #y3 = get_conv_stack(y2, 5, range(1,4), 'relu', 0.00001, 0.5)
    #y4 = GlobalMaxPool1D()(y3)
    #y5 = BatchNormalization()(y4)
    y2 = Dense(100, activation='relu')(y1)
    y = Dense(MAX_LABELS, activation='sigmoid')(y2)
    model = Model(x, y)
    adam = Adam(lr = c.lr)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=c.metrics)
    return (model, params)

def load_embeddings(embeddings_path='/home/yg2482/code/chord2vec/data/chord2vec_199.npz'):
    logger.debug('loading embeddings from: '+embeddings_path)
    global M1, M2, W, b2
    npzf = np.load(embeddings_path)
    M1 = npzf['wM1']
    M2 = npzf['wM2']
    W = npzf['wW']
    b2 = npzf['bM2']

def indices2multihot(x, r, dtype=np.float32):
    v = np.zeros(r, dtype=dtype)
    # x should belong to [1, 88]
    x = filter(lambda x: x>0, x)
    x = filter(lambda x: x<=r, x)
    # decrease x to make in [0, 87]
    x = map(lambda x: int(x-1), x)
    v[x] = 1
    return v

def square3D(x, maxlen=None, dtype=np.float32):
    if maxlen is None:
        maxlen = []
        maxlen.append(len(x))
        maxlen.append(max([0]+[len(song) for song in x]))
        maxlen.append(max([0]+[max([0]+[len(chord) for chord in song]) for song in x]))

    x_np = np.zeros(maxlen, dtype=dtype)

    for i in range(maxlen[0]):
        for j in range(maxlen[1]):
            for k in range(maxlen[2]):
                try:
                    x_np[i][j][k] = x[i][j][k]
                except IndexError:
                    break
    return x_np

def multihot3D(x, r, maxlen=None, dtype=np.float32):
    f1D = lambda chord: indices2multihot(chord,r,dtype)
    f2D = lambda song:map(f1D, song)
    x_mh = map(f2D, x)
    return x_mh
    # return square3D(x_mh, maxlen=maxlen, dtype=dtype)



def load_data(x_datapath='data/X.pickle', y_datapath='data/y.pickle', cut=1.0):
    '''
    x_datapath : path for X.pickle
    y_datapath : path for y.pickle
    cut : fraction in [0.0, 1.0] to load less data if required.
    '''
    global data, train, test, valid, MAX_CHORDS
    global labels, y_train, y_test, y_valid, MAX_LABELS, index2label, labels2index
    global train_weights
    logger.debug('loading data from: '+x_datapath)
    data = cPickle.load(open(x_datapath))
    train, test, valid = data['train'], data['test'], data['valid']
    if(cut<1.0):
        cutf = lambda x, c: x[:int(len(x)*cut)]
        train = cutf(train, cut)
        valid = cutf(valid, cut)
        test = cutf(test, cut)
        data2 = {'train':train, 'valid':valid, 'test':test}
        cPickle.dump(data2, open(x_datapath+str(cut)+'.pickle', 'w'))

    train = multihot3D(train, NUM_NOTES)
    test  = multihot3D(test, NUM_NOTES)
    valid = multihot3D(valid, NUM_NOTES)
    maxlen2D = lambda x : max([len(s) for s in x])
    MAX_CHORDS = max( map(maxlen2D, [train, test, valid]))
    # train = sequence.pad_sequences(train, MAX_CHORDS)
    # test = sequence.pad_sequences(test, MAX_CHORDS)
    # valid = sequence.pad_sequences(valid, MAX_CHORDS)

    logger.debug('loading labels from: '+y_datapath)
    labels = cPickle.load(open(y_datapath))
    if(cut<1.0):
        cutf = lambda x, c: x[:int(len(x)*cut)]
        train = cutf(labels['train'], cut)
        valid = cutf(labels['valid'], cut)
        test = cutf( labels['test'], cut)
        labels2 = {'train':train, 'valid':valid, 'test':test}
        cPickle.dump(labels2, open(y_datapath+str(cut)+'.pickle', 'w'))

    s = set()
    for k,v in labels.iteritems():
        for y in v:
            s.add(y)
    l = list(enumerate(s))
    _index2label = {k:v for k,v in l}
    index2label =  lambda x : _index2label[x]
    _labels2index = {v:k for k,v in l}
    labels2index = lambda x : _labels2index[x]

    MAX_LABELS = len(_labels2index)

    y_train = to_categorical(map(labels2index, labels['train']), MAX_LABELS)
    y_test = to_categorical(map(labels2index, labels['test']), MAX_LABELS)
    y_valid = to_categorical(map(labels2index, labels['valid']), MAX_LABELS)
    unique, counts = np.unique(np.argmax(y_train,axis=1), return_counts=True)
    #counts = np.sqrt(counts)
    train_weights=dict(zip(unique, np.divide(np.sum(counts),counts.astype('float32'))))

class DataManager():
    def __init__(self, inputs, targets, batch_size=128, maxepochs=10, transforms=lambda x:x):
        self.datasize = len(inputs)
        assert self.datasize == len(targets), 'size of targets should be the same as inputs'
        self.inputs = inputs
        self.targets = targets
        self.batch_size = batch_size
        self.maxepochs = maxepochs
        self.num_batches = int(math.ceil(float(self.datasize)/batch_size))
        if(callable(transforms)):
            transforms = [transforms, transforms]
        assert type(transforms)==list, 'transforms should be a *callable* or *list* of two callables'
        assert len(transforms)==2, 'transforms should be a callable or list of *two* callables'
        assert callable(transforms[0]) & callable(transforms[0]), 'transforms should be a callable or list of two *callables*'
        self.inputs_transform = transforms[0]
        self.targets_transform = transforms[1]
        logger.info('created a DataManager for batch_size: {}, maxepochs: {}, num_batches: {}'.format(batch_size, maxepochs, self.num_batches))

    def batch_generator(self):
        for epoch in range(self.maxepochs):
            for i in range(self.num_batches):
                logger.debug('loading batch {} of {}, epoch {}'.format(i, self.num_batches, epoch))
                start = i*self.batch_size
                end = (i+1)*self.batch_size
                inputs_batch =  self.inputs_transform(self.inputs[start:end])
                targets_batch =  self.targets_transform(self.targets[start:end])
                yield (inputs_batch, targets_batch)

def save_history(history, dirpath):
    with open(dirpath+'/training.json', 'w') as f:
        json.dump(history.params, f, indent=2)

    df = pd.DataFrame.from_dict(history.history)
    df.to_csv(dirpath+'/history.csv')
    i = df.loc[:, c.monitor].argmax()

    for m in c.metrics + ['loss']:
        plot_metric(df, m, i, dirpath)

    return

def norm_pad(x, MAX_CHORDS):
    '''
    data is a list of lists of numpy arrays output of multihot3D
    '''
    pad = sequence.pad_sequences(x, MAX_CHORDS)
    norm_pad = np.divide(pad, np.maximum(np.sum(pad, axis=2),1).reshape((pad.shape[0], pad.shape[1], 1)).astype('float32'))
    return norm_pad

def run_experiment(**kwargs):
    model, params = get_model( kwargs['embeddings'] )
    hyperparams = fill_dict(params, kwargs)

    transforms = [lambda x:norm_pad(x, MAX_CHORDS), lambda y:y]
    dm_train = DataManager(train, y_train, batch_size=c.batch_size, maxepochs=c.epochs+1, transforms=transforms)
    dm_valid = DataManager(valid, y_valid, batch_size=c.batch_size, maxepochs=100*c.epochs+1, transforms=transforms)

    with get_archiver(datadir='data/models') as a1, get_archiver(datadir='data/results') as a:

        with open(a.getFilePath('hyperparameters.json'), 'w') as f:
            json.dump(hyperparams, f, indent=2)

        with open(a.getFilePath('model.json'), 'w') as f:
            f.write(model.to_json(indent=2))

        stdout = sys.stdout
        with open(a.getFilePath('summary.txt'), 'w') as sys.stdout:
            model.summary()
        sys.stdout = stdout

        plot_model(model, to_file=a.getFilePath('model.png'), show_shapes=True, show_layer_names=True)

        earlystopping = EarlyStopping(monitor=c.monitor, patience=c.patience, verbose=0, mode=c.monitor_objective)
        modelpath = a1.getFilePath('weights.h5')
        csvlogger = CSVLogger(a.getFilePath('logger.csv'))
        modelcheckpoint = ModelCheckpoint(modelpath, monitor=c.monitor, save_best_only=True, verbose=0, mode=c.monitor_objective)
        logger.info('starting training')
        logger.info(str((dm_train.num_batches, dm_valid.num_batches)))
        h = model.fit_generator(generator=dm_train.batch_generator(), steps_per_epoch=dm_train.num_batches, epochs=c.epochs,
                        validation_data=dm_valid.batch_generator(), validation_steps=dm_valid.num_batches,
                        callbacks=[earlystopping, modelcheckpoint, csvlogger], class_weight=train_weights)

        save_history(h, a.getDirPath())

def pred(trial_ts='20170505_123642', x_datapath='../data/X.pickle', y_datapath='../data/y.pickle', model_folder='data', save=None):
    if not os.path.exists(x_datapath) or not os.path.exists(y_datapath):
        print("data file doesn't exist")
        return
    model_json_file = os.path.join(model_folder,'results/archive/', trial_ts + '_model.json')
    model_weights = os.path.join(model_folder, 'models/archive/', trial_ts + '_weights.h5')
    if os.path.exists(model_json_file) and os.path.exists(model_weights):
        model = model_from_json(open(model_json_file, 'r').read())
        model.load_weights(model_weights, by_name=False)
    else:
        print("model json or weights do not exist")
        return

    load_data(x_datapath=x_datapath, y_datapath=y_datapath)
    #MAX_CHORDS=5112
    transforms = [lambda x:norm_pad(x, MAX_CHORDS), lambda y:y]
    dm_pred = DataManager(test, y_test, batch_size=c.batch_size, transforms=transforms)
    soft = model.predict_generator(generator=dm_pred.batch_generator(), steps=dm_pred.num_batches, verbose=1)
    pred = np.argmax(soft, axis=1)
    true = np.argmax(y_test, axis=1)
    if save is not None:
        with open(save, "w") as f:
            f.write("pred,true\n")
            for (p,t)in zip(pred,true):
                f.write(str(p)+","+str(t)+"\n")

    cm = confusion_matrix(true, pred)
    print cm


def main():
    commit_hash = save_code()
    embeddings_path = '/home/yg2482/code/chord2vec/data/chord2vec_199.npz'
    x_datapath='data/X.001.pickle'
    y_datapath='data/y.001.pickle'
    load_embeddings(embeddings_path=embeddings_path)
    load_data(x_datapath=x_datapath, y_datapath=y_datapath)
    run_experiment(**locals())

if __name__ == '__main__':
    main()
    #pred()
