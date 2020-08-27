import os
import csv
import numpy as np

import matplotlib.pyplot as plt
plt.switch_backend('agg')

from keras.models import *

from keras import metrics
from keras.layers import *
from keras import optimizers

from keras.preprocessing import text
from keras.utils import to_categorical
from keras.preprocessing import sequence

from keras.models import Model
from keras.callbacks import ModelCheckpoint

    
def train_save_model():

    dic = 3

    #------------------------------------------------------------------------------------------------------------------------------
    # calculate the length of the files..

    #subtract 1 if headers are present..
    num_train = len(open('data/raw/Train.csv', 'r').readlines())
    num_test = len(open('data/raw/Test.csv', 'r').readlines())

    print('\nDataset statistics : ' + '  num_train : ' + str(num_train) + ',  num_test  : ' + str(num_test) + '\n')

    #-------------------------------------------------------------------------------------------------------
    #Loading lists..

    train_ans, anslist = [], []

    def ans_vec(data):

        f = open('data/raw/' + data + '.csv')

        lines = csv.reader(f)
        #next(lines) #to skip the header of the csv

        for line in lines:

            source_uri = line[4]

            anslist.append(source_uri)
            if data == 'Train':
                train_ans.append(source_uri)

        f.close()

    ans_vec('Train')

    #-------------------------------------------------------------------------------------------------------
    #Loading features..

    train_title_feature = np.load('data/vectorized/Train_title.npy')
    train_summary_feature = np.load('data/vectorized/Train_summary.npy')

    tokenizer_a = text.Tokenizer(num_words=dic+1)
    tokenizer_a.fit_on_texts(anslist)

    trainans_feature = tokenizer_a.texts_to_sequences(train_ans)
    trainans_feature = sequence.pad_sequences(trainans_feature, dic, padding='post', value=0, truncating='post')
    trainans_hot = to_categorical(trainans_feature, dic+1)  #one-hot

    #-------------------------------------------------------------------------------------------------------
    # model building..

    print('\nBuilding model...\n')

    #title model..
    encode_title = Input(shape=(1,128,))


    #Summary model..
    encode_summary =Input(shape=(1,128,))


    #Merge model..
    merge_model = Concatenate()([encode_title, encode_summary])
    merge_model = Activation('tanh')(merge_model)
    batch_model = BatchNormalization()(merge_model)
    
    batch_model = Dense(dic)(batch_model)
    batch_model = Permute((2, 1))(batch_model)

    #Output model..
    output_model = Dense(dic+1, activation='softmax')(batch_model)
    gate_model = Model(inputs=[encode_title, encode_summary], outputs=output_model)
    gate_model.summary()

    #Compile model..
    nadam = optimizers.Nadam()
    gate_model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=[metrics.categorical_accuracy])

    #save model..
    filepath = 'models/MODEL.hdf5'
    checkpoint = ModelCheckpoint(filepath,verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    history = gate_model.fit([ train_title_feature,train_summary_feature], trainans_hot, epochs=100, batch_size=128, callbacks=callbacks_list, verbose=1)

     

    # serialize model to JSON
    model_json = gate_model.to_json()
    with open('models/MODEL.json', "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    gate_model.save_weights('models/MODEL.h5')

    print("\nSaved model to disk...\n")

def main():

    if os.path.exists('models') == False:
        os.mkdir('models')

    if os.path.isfile('models/MODEL.hdf5') == False:

        print('\n\nTraining model...')
        train_save_model()

    print('\nTraining complete...\n\n')

if __name__ == "__main__":
    main()