#Answer Generation
import csv
import os
import numpy as np
from keras.models import *
from keras.models import Model
from keras.preprocessing import text

def load_model():
    print('\nLoading model...')  
    # load json and create model
    json_file = open('models/MODEL.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    gate_model = model_from_json(loaded_model_json)
    # load weights into new model
    gate_model.load_weights('models/MODEL.h5', by_name=True) 

    return gate_model

train_ans, anslist = [], []

def ans_vec():

    anslist = []

    dataset = ['Train']

    for data in dataset:
        f = open('data/raw/' + data + '.csv')

        lines = csv.reader(f)

        for line in lines:
            source_uri = line[4]
            anslist.append(source_uri)

        f.close()
    
    return anslist


def generate_save_ans():

    dic = 3

    anslist = ans_vec()
    gate_model = load_model()

    test_title_feature = np.load('data/vectorized/Test_title.npy')
    test_summary_feature = np.load('data/vectorized/Test_summary.npy')

    tokenizer_a = text.Tokenizer(num_words=dic+1)
    tokenizer_a.fit_on_texts(anslist)

    dic_a = tokenizer_a.word_index
    ind_a ={value:key for key, value in dic_a.items()}

    num_test = len(open('data/raw/Test.csv', 'r').readlines())
    
    ans = gate_model.predict([ test_title_feature, test_summary_feature])
    fp = open('reports/Test.ans', 'w')
    for h in range(num_test):
        i = h
        
        if np.argmax(ans[i][0],axis=0) == 0:
            fp.write('indiatimes\n')  #Low frequency words are replaced with "indiatimes"
        else:
            for j in range(dic):
                an = np.argmax(ans[i][j],axis=0)
                if j != dic-1:
                    anext = np.argmax(ans[i][j+1],axis=0)
                    if an != 0 and anext != 0:  #Words before and after
                        if an == anext:
                            fp.write('')  #Delete duplicate words
                        else:
                            fp.write(ind_a[an] + ' ')
                    elif an != 0 and anext == 0:
                        fp.write(ind_a[an])
                    elif an == 0 and anext != 0: 
                        fp.write(ind_a[anext])
                    else:
                        fp.write('')
                else:
                    if an != 0:
                        fp.write(ind_a[an] + '\n')
                    else:
                        fp.write('\n')
    fp.close()

def main():
    
    load_model()
    print('\n\nGenerating answers...')

    if os.path.exists('reports') == False:
        os.mkdir('reports')

    if os.path.isfile('reports/Test.ans') == False:
        generate_save_ans()

    print('\nAnswer generation complete...\n\n')

if __name__ == "__main__":
    main()