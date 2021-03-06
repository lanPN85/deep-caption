from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential
from keras.layers import Flatten, LSTM, Dense, TimeDistributed
from .layers import DecoderLSTM


def vgg_imgnet_decode(caption_model, weights='imagenet'):
    model = Sequential()
    base = VGG16(include_top=False, input_shape=(224, 224, 3),
                 weights=weights)
    model.add(base)
    model.add(Flatten())
    model.add(Dense(caption_model.connector_dim,
                    activation='relu'))
    # depth = int(caption_model.connector_dim / caption_model.vocab.size)
    # model.add(Reshape((caption_model.vocab.size, depth)))
    model.add(DecoderLSTM(None,
                          caption_model.connector_dim,
                          caption_model.sentence_len,
                          dropout=caption_model.dropout,
                          recurrent_dropout=caption_model.dropout,
                          activation='tanh'))
    model.add(LSTM(caption_model.connector_dim,
                   dropout=caption_model.dropout,
                   recurrent_dropout=caption_model.dropout,
                   activation='tanh', return_sequences=True
                   ))
    model.add(TimeDistributed(Dense(caption_model.vocab.size,
                                    activation='softmax')))

    return model


def resnet_imgnet_decode(caption_model, weights='imagenet'):
    model = Sequential()
    base = ResNet50(include_top=False, input_shape=(224, 224, 3),
                    weights=weights)
    model.add(base)
    model.add(Flatten())
    model.add(Dense(caption_model.connector_dim,
                    activation='relu'))
    # depth = int(caption_model.connector_dim / caption_model.vocab.size)
    # model.add(Reshape((caption_model.vocab.size, depth)))
    model.add(DecoderLSTM(None,
                          caption_model.connector_dim,
                          caption_model.sentence_len,
                          dropout=caption_model.dropout,
                          recurrent_dropout=caption_model.dropout,
                          activation='tanh'))
    model.add(LSTM(caption_model.connector_dim,
                   dropout=caption_model.dropout,
                   recurrent_dropout=caption_model.dropout,
                   activation='tanh', return_sequences=True
                   ))
    model.add(TimeDistributed(Dense(caption_model.vocab.size,
                                    activation='softmax')))

    return model
