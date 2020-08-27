#Text Vectorization
import os
import pandas
import numpy as np

#used for sentence embedding
import tensorflow as tf 
import tensorflow_hub as hub


def text_vectorization(data):

    module_url = "https://tfhub.dev/google/nnlm-en-dim128/2"
    embed = hub.KerasLayer(module_url)

    colnames = ['event_uri','title','date','summary','source_uri']
    df = pandas.read_csv('data/raw/' + data + '.csv', names=colnames)

    #Creating a list of titles and summaries..
    lst_title = df.title.tolist()
    lst_summary = df.summary.tolist()
    
    #Vectorization of titles..
    embeddings_title = embed(lst_title)
    embeddings_title = tf.make_tensor_proto(embeddings_title)
    embeddings_title = tf.make_ndarray(embeddings_title)

    embeddings_title = np.array(embeddings_title)
    embeddings_title = np.expand_dims(embeddings_title, axis=1)
    #embeddings_title = np.squeeze(embeddings_title)  
    np.save('data/vectorized/' + data + '_title.npy', embeddings_title) 

    #Vectorization of summaries..
    embeddings_summary = embed(lst_summary)
    embeddings_summary = tf.make_tensor_proto(embeddings_summary)
    embeddings_summary = tf.make_ndarray(embeddings_summary)

    embeddings_summary = np.array(embeddings_summary)
    embeddings_summary = np.expand_dims(embeddings_summary, axis=1)
    #embeddings_summary = np.squeeze(embeddings_summary)  
    np.save('data/vectorized/' + data + '_summary.npy', embeddings_summary)


def main():

    print('\n\nTurning text into vectors...')

    if os.path.exists('data/vectorized') == False:
        os.mkdir('data/vectorized')

    if os.path.isfile('data/vectorized/Test_title.npy') == False:

        text_vectorization('Train')
        text_vectorization('Test')
    
    print('\nVectorization complete...\n\n')

if __name__ == "__main__":
    main()