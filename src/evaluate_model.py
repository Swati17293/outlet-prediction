#Answer Evaluation
import csv
import numpy as np

from sklearn.metrics import accuracy_score 
from sklearn.metrics import hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.dummy import DummyClassifier 
import warnings

def evaluate():

    warnings.filterwarnings("ignore", category=UserWarning)
    
    f_test = open('data/raw/Test.csv')

    lines_test = csv.reader(f_test)

    f_train = open('data/raw/Train.csv')

    lines_train = csv.reader(f_train)

    #next(lines) #to skip the header of the csv

    true_ans_test = []
    true_ans_train = []

    train_title_feature = np.load('data/vectorized/Train_title.npy')
    train_summary_feature = np.load('data/vectorized/Train_summary.npy')

    test_title_feature = np.load('data/vectorized/Test_title.npy')
    test_summary_feature = np.load('data/vectorized/Test_summary.npy')

    for line in lines_test:
        source_uri = line[4]
        true_ans_test.append(source_uri.split(' '))

    for line in lines_train:
        source_uri = line[4]
        true_ans_train.append(source_uri.split(' '))


    f = open('reports/Test.ans')

    lines = csv.reader(f)
    #next(lines) #to skip the header of the csv

    pred_ans = []

    for line in lines:
        pred_ans.append(line[0].split(' '))

    f.close()

    classes = ['nytimes', 'indiatimes', 'washingtonpost']
    mlb = MultiLabelBinarizer(classes)
    pred_ans_b = mlb.fit_transform(pred_ans)
    true_ans_b = mlb.transform(true_ans_test)

    print('\n\nMLB:')

    Sub_accuracy_score = accuracy_score(true_ans_b, pred_ans_b)
    Sub_accuracy_score = str(round(Sub_accuracy_score, 3))

    print('\nSubset Accuracy: ' + Sub_accuracy_score)

    hamming_score = hamming_loss(true_ans_b, pred_ans_b)
    hamming_score = str(round(hamming_score, 3))

    print('\nHamming Loss: ' + hamming_score + '\n\n')

    strategies = ['stratified', 'uniform'] 

    X_test = np.squeeze(np.concatenate((test_title_feature, test_summary_feature), 2))
    y_test = true_ans_test

    X_train = np.squeeze(np.concatenate((train_title_feature, train_summary_feature), 2))
    y_train = true_ans_train
    
    test_scores = [] 
    for s in strategies: 
        
        dclf = DummyClassifier(strategy = s, random_state = 0) 
        dclf = dclf.fit(X_train, y_train) 

        pred_ans = []

        ans = dclf.predict(X_test)

        for a in ans:
            pred_ans.append(a)

        pred_ans_b = mlb.fit_transform(pred_ans)

        print('\n\n' + s + ':')

        Sub_accuracy_score = accuracy_score(true_ans_b, pred_ans_b)
        Sub_accuracy_score = str(round(Sub_accuracy_score, 3))

        print('\nSubset Accuracy: ' + Sub_accuracy_score)

        hamming_score = hamming_loss(true_ans_b, pred_ans_b)
        hamming_score = str(round(hamming_score, 3))

        print('\nHamming Loss: ' + hamming_score)

    print('\n\n')

def main():
    evaluate()

if __name__ == "__main__":
    main()