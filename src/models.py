import logging
logging.basicConfig(level = logging.DEBUG, format=
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
import random
from collections import Counter

import keras
from keras.layers import Input, Embedding, Conv1D, GlobalMaxPool1D, Dense, GlobalAvgPool1D, Dropout, BatchNormalization, Flatten, Activation
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

from conf_mat import save_conf_mat

MAX_CHORDS = None
MAX_LABELS = None
# MAX_CHORDS = 5587
# MAX_LABELS = 13
NUM_NOTES = 88
NUM_DIM = 1024

M1 = M2 = W = b2 = None
data= train= test= valid= MAX_CHORDS = None
labels= y_train= y_test= y_valid= MAX_LABELS= index2label= labels2index = None
train_weights= None

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

def get_model(use_embeddings=True, dilated_convs=False):
    params = {k:v for k,v in locals().iteritems() if k!='weights'}
    x = Input(shape=(MAX_CHORDS,NUM_NOTES), dtype='float32')
    if use_embeddings:
        y1 = Dense(NUM_DIM, activation='linear', use_bias=False, weights=[M1], trainable=False)(x)
    else:
        y1 = x
    #y2 = BatchNormalization()(y1)
    #y3 = get_conv_stack(y2, 5, range(1,4), 'relu', 0.00001, 0.5)
    #y2 = GlobalMaxPool1D()(y1)
    #y5 = BatchNormalization()(y4)
    y2 = Flatten()(y1)
    y3 = Dropout(0.5)(y2)
    y4 = Dense(100)(y3)
    y5 = BatchNormalization()(y4)
    y6 = Activation('relu')(y5)
    y7 = Dropout(0.5)(y6)
    y = Dense(MAX_LABELS, activation='sigmoid')(y7)
    model = Model(x, y)
    adam = Adam(lr=c.lr)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=c.metrics)
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

def filter_data_by_index(data, idx):
    return [x for i,x in enumerate(data) if idx[i]]
    
def filter_index(X,y,idx):    
    idxs = map(lambda x:x[idx]!=1,y)
    filter_indices = lambda data : [x for i,x in enumerate(data) if idxs[i]]
    return filter_indices(X), filter_indices(y)



def filter_majority_index():
    global train, y_train, test, y_test, valid, y_valid
    most_common_index = np.argmax(y_train.sum(axis=0))
    (train, y_train) = filter_index(train, y_train, most_common_index)
    (test, y_test) = filter_index(test, y_test, most_common_index)
    (valid, y_valid) = filter_index(valid, y_valid, most_common_index)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_valid = np.array(y_valid)
    

import random

def shuffle_train_valid(xt, yt, xv, yv):
    l1 = len(yt)
    l2 = len(yv)
    assert (l1 == len(xt))
    assert (l2 == len(xv))
    c = zip(xt+xv, yt+yv)
    random.shuffle(c)
    xtv, ytv = zip(*c)
    xt = xtv[:l1]
    xv = xtv[l1:]
    yt = ytv[:l1]
    yv = ytv[l1:]    
    return (xt, yt, xv, yv)

def log_about_data():
    logger.debug( 'train:\n'+util.about(train, SINGLE_LINE=True) )
    logger.debug( 'valid:\n'+util.about(valid, SINGLE_LINE=True) )
    logger.debug( 'test:\n'+util.about(test, SINGLE_LINE=True) )
    logger.debug( 'y_train:\n'+util.about(y_train, SINGLE_LINE=True) )
    logger.debug( 'y_valid:\n'+util.about(y_valid, SINGLE_LINE=True) )
    logger.debug( 'y_test:\n'+util.about(y_test, SINGLE_LINE=True) )

def get_balanced_data_index(y, classes):
    c1 = Counter()
    for l in y:
        if l in classes:
            c1[l]+=1
    target = min( c1.values() )
    logger.debug(str(classes))
    logger.debug(str(c1))
    logger.debug(str(target))
    
    for k in c1.keys():
        c1[k] = target

    logger.debug(str(c1))

    index = [False]*len(y)
    for i in range(len(y)):
        if(y[i] in c1.keys()):
            if(c1[y[i]] > 0):
                index[i] = True
                c1[y[i]]-=1
    return index
 
    
def load_data(x_datapath='data/X.pickle', y_datapath='data/y.pickle', load_train=True):
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

    logger.debug('loading labels from: '+y_datapath)
    labels = cPickle.load(open(y_datapath))

    logging.debug('shuffling train and valid data and labels')
    (data['train'], labels['train'], data['valid'], labels['valid'] ) = \
        shuffle_train_valid( data['train'], labels['train'], data['valid'], labels['valid'] )

    s = Counter()
    for k,v in labels.iteritems():
        for y in v:
            s[y]+=1

    l = list(enumerate(s.keys()))
    _index2label = {k:v for k,v in l}
    index2label =  lambda x : _index2label[x]
    _labels2index = {v:k for k,v in l}
    labels2index = lambda x : _labels2index[x]

    two_most_common = map( lambda x:x[0], s.most_common(n=2))
    logging.info('two most common: ' + str(s.most_common(n=2)))
    logging.info('two most common: ' + str(two_most_common))
    
    index_train = get_balanced_data_index(labels['train'], classes=two_most_common)
    index_valid = get_balanced_data_index(labels['valid'], classes=two_most_common)
    index_test = get_balanced_data_index(labels['test'], classes=two_most_common)
    
    logger.info('index_train average: ' + str(np.average(index_train)))
    logger.info('valid_train average: ' + str(np.average(index_valid)))
    logger.info('test_train average: ' + str(np.average(index_test)))
    
    logger.info('converting to multihot')
    
    train = multihot3D(filter_data_by_index(data['train'], index_train), NUM_NOTES) if load_train else None
    valid = multihot3D(filter_data_by_index(data['valid'], index_valid), NUM_NOTES)
    test = multihot3D(filter_data_by_index(data['test'], index_test), NUM_NOTES)
    
    maxlen2D = lambda x : max([len(s) for s in x])
    MAX_CHORDS = max( map(maxlen2D, [train, test, valid]))
    
    y_train = to_categorical(map(labels2index, filter_data_by_index(labels['train'], index_train)), MAX_LABELS)
    y_test = to_categorical(map(labels2index, filter_data_by_index(labels['test'], index_test)), MAX_LABELS)
    y_valid = to_categorical(map(labels2index, filter_data_by_index(labels['valid'], index_valid)), MAX_LABELS)

    log_about_data()
    
    return

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

    # pred, true = pred(trial_ts=trial_ts, x_datapath=x_datapath, y_datapath=y_datapath, model_folder=model_folder, save=save)


    return

def norm_pad(x, MAX_CHORDS):
    '''
    data is a list of lists of numpy arrays output of multihot3D
    '''
    pad = sequence.pad_sequences(x, MAX_CHORDS)
    norm_pad = np.divide(pad, np.maximum(np.sum(pad, axis=2),1).reshape((pad.shape[0], pad.shape[1], 1)).astype('float32'))
    return norm_pad

def run_experiment(**kwargs):
    model, params = get_model( kwargs['use_embeddings'] )
    hyperparams = fill_dict(params, kwargs)

    transforms = [lambda x:norm_pad(x, MAX_CHORDS), lambda y:y]
    dm_train = DataManager(train, y_train, batch_size=c.batch_size, maxepochs=100*c.epochs+1, transforms=transforms)
    dm_valid = DataManager(valid, y_valid, batch_size=c.batch_size, maxepochs=100*c.epochs+1, transforms=transforms)
    dm_test = DataManager(test, y_test, batch_size=c.batch_size, maxepochs=100*c.epochs+1, transforms=transforms)

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
        logger.info('ending training')
        save_history(h, a.getDirPath())

        logger.info('creating confusion matrix')
        soft = model.predict_generator(generator=dm_test.batch_generator(), steps=dm_test.num_batches, verbose=1)
        pred = np.argmax(soft, axis=1)
        true = np.argmax(y_test, axis=1)
        try:
            cm = save_conf_mat(true, pred, a.getDirPath())
        except Exception as e:
            logger.exception(e)
 

def main():
    commit_hash = save_code()
    embeddings_path = 'data2/chord2vec_30hr.npz'
    x_datapath='data2/X.001.pickle'
    y_datapath='data2/y.001.pickle'
    load_embeddings(embeddings_path=embeddings_path)
    load_data(x_datapath=x_datapath, y_datapath=y_datapath)
    use_embeddings = True
    run_experiment(**locals())

if __name__ == '__main__':
    main()
    #pred()

