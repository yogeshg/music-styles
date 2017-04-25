import logging
logging.basicConfig(level = logging.DEBUG, format=
        '%(asctime)s:%(levelname)s:%(name)s:%(threadName)s:line %(lineno)d: %(message)s')
logger = logging.getLogger(__name__)

import cPickle
import numpy as np
import math
import keras
from keras.layers import Input, Embedding, Conv1D, GlobalMaxPool1D, Dense, GlobalAvgPool1D, Dropout
from keras.layers import concatenate
from keras.models import Model
from keras.preprocessing import sequence
from keras import regularizers
from keras.engine.topology import Layer
from keras.utils import to_categorical
from keras.optimizers import Adam

# MAX_CHORDS = 150
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

def get_model():
    params = {k:v for k,v in locals().iteritems() if k!='weights'}
    # x = Input(shape=(NUM_NOTES,), dtype='float32')
    # TODO: NORMALIZE!!!
    x = Input(shape=(MAX_CHORDS,NUM_NOTES), dtype='float32')
    y1 = Dense(NUM_DIM, activation='linear', use_bias=False, weights=[M1], trainable=False)(x)
    y2 = get_conv_stack(y1, 5, range(1,4), 'relu', 0.00001, 0.5)
    y3 = GlobalMaxPool1D()(y2)
    y = Dense(MAX_LABELS, activation='sigmoid')(y3)
    model = Model(x, y)
    adam = Adam(lr = 0.0001)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
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

def load_data(x_datapath='/home/yg2482/code/chord2vec/data/X.pickle',
        y_datapath='/home/yg2482/code/chord2vec/data/y.pickle'):
    global data, train, test, valid, MAX_CHORDS
    global labels, y_train, y_test, y_valid, MAX_LABELS, index2label, labels2index
    logger.debug('loading data from: '+x_datapath)
    data = cPickle.load(open(x_datapath))
    train, test, valid = data['train'], data['test'], data['valid']
    train = multihot3D(train, NUM_NOTES)
    test  = multihot3D(test, NUM_NOTES)
    valid = multihot3D(valid, NUM_NOTES)
    maxlen2D = lambda x : max([len(s) for s in x])
    MAX_CHORDS = max( map(maxlen2D, [train, test, valid]))
    train = sequence.pad_sequences(train, MAX_CHORDS)
    test = sequence.pad_sequences(test, MAX_CHORDS)
    valid = sequence.pad_sequences(valid, MAX_CHORDS)
    # TODO: NORMALIZE!!!

    logger.debug('loading labels from: '+y_datapath)
    labels = cPickle.load(open(y_datapath))
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

    def batch_generator(self):
        for epoch in range(self.maxepochs):
            for i in range(self.num_batches):
                logger.debug('loading batch {} of {}, epoch {}'.format(i, self.num_batches, epoch))
                start = i*self.batch_size
                end = (i+1)*self.batch_size
                inputs_batch =  self.inputs_transform(self.inputs[start:end])
                targets_batch =  self.targets_transform(self.targets[start:end])
                yield (inputs_batch, targets_batch)
                # for (inpt, trgt) in zip(inputs_batch, targets_batch):
                #     yield (inpt, trgt)
        return

# load_data()
# load_embeddings()

