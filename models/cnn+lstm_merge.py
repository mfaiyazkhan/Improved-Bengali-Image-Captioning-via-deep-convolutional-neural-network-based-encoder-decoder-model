from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM, Embedding, Dropout, Dense
from keras.layers.merge import add


def cnn_lstm_merge(max_length=42, vocab_size=5459):
    input_1 = Input(shape=(2048,))
    dropout_1 = Dropout(0.5)(input_1)
    dense_1 = Dense(512, activation='relu')(dropout_1)

    input_2 = Input(shape=(max_length,))
    embedding_1 = Embedding(vocab_size, 512)(input_2)
    dropout_2 = Dropout(0.5)(embedding_1)
    lstm_1 = LSTM(512)(dropout_2)

    add_1 = add([dense_1, lstm_1])
    dense_2 = Dense(512, activation='relu')(add_1)
    dense_3 = Dense(vocab_size, activation='softmax')(dense_2)

    model = Model(inputs=[input_1, input_2], outputs=dense_3)
    return model


# model_ = cnn_lstm_merge(42, 5459)
# model_.summary()
