from keras.models import Model
from keras.layers import Input, Add
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Embedding, Dropout, Dense


def cnn_cnn_merge(max_length=42, vocab_size=5459):
    input_1 = Input(shape=(2048,))
    dropout_1 = Dropout(0.5)(input_1)
    dense_1 = Dense(512, activation='relu')(dropout_1)

    input_2 = Input(shape=(max_length,))
    embedding_1 = Embedding(vocab_size, 512)(input_2)
    dropout_2 = Dropout(0.5)(embedding_1)

    conv_1D = Conv1D(filters=512, kernel_size=3, activation='relu')(dropout_2)
    pool_1 = GlobalMaxPooling1D()(conv_1D)

    add_1 = Add()([dense_1, pool_1])
    dense_2 = Dense(512, activation='relu')(add_1)
    dense_3 = Dense(vocab_size, activation='softmax')(dense_2)

    model = Model(inputs=[input_1, input_2], outputs=dense_3)
    return model


# model_ = cnn_cnn_merge(42, 5459)
# model_.summary()
