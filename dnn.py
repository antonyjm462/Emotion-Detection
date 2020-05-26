
class Dnn:
    from keras.models import Sequential
    def __init__(self):
        pass


    def construct_dnn(self,num_input_features=500):
        from keras.layers import Dense,Activation,Dropout
        dnn_model = Sequential()
        dnn_model.add(Dense(512, activation='relu', input_shape=(num_input_features,)))
        dnn_model.add(Dropout(0.2))
        dnn_model.add(Dense(512, activation='relu'))
        dnn_model.add(Dropout(0.2))
        dnn_model.add(Dense(512, activation='relu'))
        dnn_model.add(Dropout(0.2))
        dnn_model.add(Dense(10))
        dnn_model.add(Activation('softmax'))

        dnn_model.compile(loss='categorical_crossentropy', optimizer='adam',                 
                          metrics=['accuracy'])
        dnn_model.summary()
        return dnn_model 


    def construct_cnn(self,embedding_len=32,total_vocab=5000,num_input_features=500):
        from keras.layers import Dense,Activation,Dropout,Conv1D,Flatten
        from keras.layers import Embedding
        model = Sequential()
        model.add(Embedding(total_vocab,embedding_len,input_shape=(num_input_features,)))
        model.add(Conv1D(128,3,padding = 'same'))
        model.add(Conv1D(64,3,padding = 'same'))
        model.add(Conv1D(32,2,padding = 'same'))
        model.add(Conv1D(16,2,padding = 'same'))
        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(100,activation = 'sigmoid'))
        model.add(Dropout(0.2))
        model.add(Dense(10,activation='sigmoid'))
        model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
        model.summary()
        return model

    def construct_LSTM(self,vocab_size=5000,max_len=500):
      from keras.layers import Dense, Embedding, Dropout, SpatialDropout1D
      from keras.layers import LSTM
      EMBEDDING_DIM = 128 # dimension for dense embeddings for each token
      LSTM_DIM = 64 # total LSTM units
      model = Sequential()
      model.add(Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=max_len))
      model.add(SpatialDropout1D(0.2))
      model.add(LSTM(LSTM_DIM, dropout=0.2, recurrent_dropout=0.2))
      model.add(Dense(1, activation="sigmoid"))
      model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])
      return model