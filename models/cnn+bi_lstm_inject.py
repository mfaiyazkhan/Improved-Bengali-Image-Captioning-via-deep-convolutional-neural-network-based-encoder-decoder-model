from keras.models import Model
from keras.layers import concatenate
from keras.layers import LSTM, Embedding, BatchNormalization, Dropout, TimeDistributed, Dense, RepeatVector
from keras.layers.wrappers import Bidirectional
from keras.models import Sequential


def cnn_bi_lstm_inject(max_length=42, vocab_size=5459):
    image_model = Sequential()
    image_model.add(Dense(512, input_shape=(2048,), activation='relu'))
    image_model.add(RepeatVector(max_length))

    lang_model = Sequential()
    lang_model.add(Embedding(vocab_size, 512, input_length=max_length))
    lang_model.add(Bidirectional(LSTM(256, return_sequences=True)))
    lang_model.add(Dropout(0.5))
    lang_model.add(BatchNormalization())
    lang_model.add(TimeDistributed(Dense(512)))

    model_concat = concatenate([image_model.output, lang_model.output], axis=-1)
    model_concat = Dropout(0.5)(model_concat)
    model_concat = BatchNormalization()(model_concat)
    model_concat = Bidirectional(LSTM(256, return_sequences=False))(model_concat)
    model_concat = Dense(vocab_size, activation='softmax')(model_concat)

    model = Model(inputs=[image_model.input, lang_model.input], outputs=model_concat)
    return model


# model_ = cnn_bi_lstm_inject(42, 5459)
# model_.summary()
