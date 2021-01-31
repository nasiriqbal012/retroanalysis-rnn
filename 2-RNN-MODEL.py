# script for Recurrent Neural Network Model (Long Short-term Memory)
import sys

try:

    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from keras.utils import np_utils
    import pickle
    from keras.optimizers import SGD
    import tensorflow as tf
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OneHotEncoder
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import LSTM
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras import layers
    from keras.utils import to_categorical
    import keras.backend as K
    from keras import backend as K
    import re
    from tensorflow import keras
    import time
    from sklearn.utils import compute_class_weight
    from sklearn.utils import class_weight
    import matplotlib.pyplot as plt
    from keras.models import load_model
    from keras.layers.convolutional import Conv1D, Conv3D
    from keras.layers.convolutional import MaxPooling1D, MaxPooling3D

    from imblearn.over_sampling import SMOTE
    from collections import Counter
    from imblearn.under_sampling import TomekLinks
    from sklearn.utils import resample
    from matplotlib import pyplot



    start_time = time.time()  # Start time of execution


    trainGraph = input('Enter train graph file address: ')
    trainGraph = str(trainGraph)
    trainLabel = input('Enter train labels file address: ')
    trainLabel = str(trainLabel)

    testGraph = input('Enter test graph file address: ')
    testGraph = str(testGraph)
    testLabel = input('Enter test labels file address: ')
    testLabel = str(testLabel)

    valGraph = input('Enter validation graph file address: ')
    valGraph = str(valGraph)
    valLabel = input('Enter validation labels file address: ')
    valLabel = str(valLabel)


    with open(trainGraph, 'rb') as filehandle:  # to read stored data
        # read the data as binary data stream
        tG = pickle.load(filehandle)
    with open(trainLabel, 'rb') as filehandle:  # to read stored data
        # read the data as binary data stream
        tL = pickle.load(filehandle)

    with open(testGraph, 'rb') as filehandle:  # to read stored data
        # read the data as binary data stream
        sG = pickle.load(filehandle)
    with open(testLabel, 'rb') as filehandle:  # to read stored data
        # read the data as binary data stream
        sL = pickle.load(filehandle)

    with open(valGraph, 'rb') as filehandle:  # to read stored data
        # read the data as binary data stream
        vG = pickle.load(filehandle)
    with open(valLabel, 'rb') as filehandle:  # to read stored data
        # read the data as binary data stream
        vL = pickle.load(filehandle)

    # to simplify data
    tG = np.array(tG)
    mx = np.max(tG)
    tG = tG / mx

    sG = np.array(sG)
    mx = np.max(sG)
    sG = sG / mx

    # to simplify data
    vG = np.array(vG)
    mx = np.max(vG)
    vG = vG / mx

    tL = np.array(tL, dtype='i')
    sL = np.array(sL, dtype='i')
    vL = np.array(vL, dtype='i')

    class_weights = { 0:0.33084552,
                      1:0.41998741,
                      2:0.88775782,
                      3:5.55958333,
                      4:7.69788462,
                      5:0.59878833,
                      6:1.09130316,
                      7:6.14884793,
                      8:2.73235495,
                      9:22.11546961}
    #print(class_weights)



    #####
    # ENCODING
    onehot_encoder = OneHotEncoder(sparse=False)
    e_tL = tL.reshape(len(tL), 1)
    encoded_tL = onehot_encoder.fit_transform(e_tL)

    onehot_encoder = OneHotEncoder(sparse=False)
    e_sL = sL.reshape(len(sL), 1)
    encoded_sL = onehot_encoder.fit_transform(e_sL)

    onehot_encoder = OneHotEncoder(sparse=False)
    e_vL = vL.reshape(len(vL), 1)
    encoded_vL = onehot_encoder.fit_transform(e_vL)

    # Splitting the data
    x_train, y_train = tG, encoded_tL
    x_test, y_test = sG, encoded_sL
    x_val, y_val = vG[:5000], encoded_vL[:5000]


    ####################################################################################################################
    ####################################################################################################################

    y_train = LabelEncoder().fit_transform(tL.ravel())
    y_test = LabelEncoder().fit_transform(sL.ravel())
    y_val = LabelEncoder().fit_transform(vL.ravel())

    # summarize distribution
    counter = Counter(y_train)
    print(counter)
    for k, v in counter.items():
        per = v / len(y_train) * 100
        print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
    # plot the distribution
    pyplot.bar(counter.keys(), counter.values())
    pyplot.show()

    print(x_train.shape)
    x_train = x_train.reshape(x_train.shape[0], -1)
    print(x_train.shape)

    # for sythesising minority classes
    oversample = SMOTE()
    x_train, y_train = oversample.fit_resample(x_train.reshape(x_train.shape[0], -1), y_train)
    # summarize distribution
    counter = Counter(y_train)
    for k, v in counter.items():
        per = v / len(y_train) * 100
        print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
    # plot the distribution
    pyplot.bar(counter.keys(), counter.values())
    pyplot.show()

    x_train = x_train.reshape(x_train.shape[0], 86, 86)

    #with open('x_train_SMOTE.data', 'ab') as filehandle:
        #store the data as binary data stream
     #   pickle.dump(x_train, filehandle)
    #filehandle.close()

    onehot_encoder = OneHotEncoder(sparse=False)
    e_tL = y_train.reshape(len(y_train), 1)
    y_train = onehot_encoder.fit_transform(e_tL)

    #######################################################################################################
    #######################################################################################################
    # Function for system evaluation
    def check_units(y_true, y_pred):
        if y_pred.shape[1] != 1:
            y_pred = y_pred[:, 1:2]
            y_true = y_true[:, 1:2]
        return y_true, y_pred

    def precision(y_true, y_pred):
        y_true, y_pred = check_units(y_true, y_pred)
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def recall(y_true, y_pred):
        y_true, y_pred = check_units(y_true, y_pred)
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def f1(y_true, y_pred):
        def recall(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        def precision(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision

        y_true, y_pred = check_units(y_true, y_pred)
        precision = precision(y_true, y_pred)
        recall = recall(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


    #######################################################################################################
    #######################################################################################################
    ### => RNNs MODEL
    #######################################################################################################

    #  initialize our recurrent neural network
    rnn = Sequential()

    # adding a convolutional layer
    #rnn.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    #rnn.add(MaxPooling1D(pool_size=2))

    # Adding Our First LSTM Layer
    rnn.add(layers.Bidirectional(LSTM(units=10, return_sequences=True, input_shape=(86, 86))))

    # Adding Some Dropout Regularization
    rnn.add(Dropout(0.2))

    # Adding More LSTM Layers With Dropout Regularization
    for i in [True, True, True, True, True, False]:
        rnn.add(layers.Bidirectional(LSTM(units=10, return_sequences=i)))
        rnn.add(Dropout(0.2))

    # Adding The Output Layer To Our Recurrent Neural Network
    rnn.add(Dense(units=1, activation='sigmoid'))

    # Compiling Our Recurrent Neural Network
    opt = SGD(lr=0.0001)
    rnn.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy', precision, recall, f1])

    # Fitting The Recurrent Neural Network On The Training Set
    history = rnn.fit( x_train, y_train, epochs=2, batch_size=1000, validation_data=(x_test, y_test))


    #rnn.save('lstm_model_01.h5')
    train_loss, train_accuracy = rnn.evaluate(x_train, y_train, verbose=0)
    print ('Train loss: {}, Train accuracy: {}'.format(train_loss, train_accuracy*100))

    test_loss, test_accuracy = rnn.evaluate(x_test, y_test, verbose=0)
    print ('Test loss: {}, Test accuracy: {}'.format(test_loss, test_accuracy*100))

    val_loss, val_accuracy = rnn.evaluate(x_val, y_val, verbose=0)
    print('Validation loss: {}, Validation accuracy: {}'.format(val_loss, val_accuracy * 100))

    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    plt.plot(history.history['precision'])
    plt.plot(history.history['val_precision'])
    plt.title('model precision')
    plt.ylabel('precision')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['recall'])
    plt.plot(history.history['val_recall'])
    plt.title('model recall')
    plt.ylabel('recall')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['f1'])
    plt.plot(history.history['val_f1'])
    plt.title('model f1-score')
    plt.ylabel('f1-score')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    print (history)

    print("--- %s seconds ---" % (time.time() - start_time))

except Exception as e:
    print(e)
finally:
    print("Terminating this code..!")

