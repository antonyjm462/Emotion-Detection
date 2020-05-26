from keras.layers import Embedding

class Dnn:
    def __init__(self):
        pass


    def construct_dnn(self,num_input_features):
        dnn_model = Sequential()
        dnn_model.add(Dense(512, activation='relu', input_shape=(num_input_features,)))
        dnn_model.add(Dropout(0.2))
        dnn_model.add(Dense(512, activation='relu'))
        dnn_model.add(Dropout(0.2))
        dnn_model.add(Dense(512, activation='relu'))
        dnn_model.add(Dropout(0.2))
        dnn_model.add(Dense(2))
        dnn_model.add(Activation('softmax'))

        dnn_model.compile(loss='categorical_crossentropy', optimizer='adam',                 
                          metrics=['accuracy'])
        dnn_model.summary()
        return dnn_model 


    def construct_cnn(self,embedding_len=32,total_vocab=5000,upper_threshold=256):
        model = Sequential()
        model.add(Embedding(total_vocab,embedding_len,input_length = upper_threshold))
        model.add(Conv1D(128,3,padding = 'same'))
        model.add(Conv1D(64,3,padding = 'same'))
        model.add(Conv1D(32,2,padding = 'same'))
        model.add(Conv1D(16,2,padding = 'same'))
        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(100,activation = 'sigmoid'))
        model.add(Dropout(0.2))
        model.add(Dense(1,activation='sigmoid'))
        model.summary()
        return model