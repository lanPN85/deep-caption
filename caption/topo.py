from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Flatten, Reshape, Dense, TimeDistributed
from .layers import DecoderLSTM

from .topologies import LSTM1


def vgg_imgnet_decode(caption_model):
    model = Sequential()
    base = VGG16(include_top=False, input_shape=(224, 224, 3),
                 weights='imagenet')
    model.add(base)
    model.add(Flatten())
    model.add(Dense(caption_model.connector_dim,
                    activation='relu'))
    # depth = int(caption_model.connector_dim / caption_model.vocab.size)
    # model.add(Reshape((caption_model.vocab.size, depth)))
    model.add(DecoderLSTM(caption_model.connector_dim,
                          caption_model.sentence_len,
                          dropout=caption_model.dropout,
                          recurrent_dropout=caption_model.dropout,
                          activation='tanh'))
    model.add(TimeDistributed(Dense(caption_model.vocab.size,
                                    activation='softmax')))

    return model


def vgg_imgnet_full(caption_model):
    model = Sequential()
    base = VGG16(include_top=False, input_shape=(224, 224, 3))
    model.add(base)
    model.add(Flatten())
    model.add(Dense(caption_model.connector_dim))

    return model