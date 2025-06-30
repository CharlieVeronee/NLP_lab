#download word vectors
from urllib.request import urlretrieve
import os
import h5py
import numpy as np

def load_ConceptNet():
    os.makedirs('datasets', exist_ok=True) #create datasets directory

    if not os.path.isfile('datasets/mini.h5'):
        print("Downloading Conceptnet Numberbatch word embeddings...")
        conceptnet_url = 'http://conceptnet.s3.amazonaws.com/precomputed-data/2016/numberbatch/17.06/mini.h5'
        urlretrieve(conceptnet_url, 'datasets/mini.h5')


#extract embeddings from mini.h5 dataset
def extract_embeddings():
    #load the file and pull out words and embeddings
    with h5py.File('datasets/mini.h5', 'r') as f:
        all_words = [word.decode('utf-8') for word in f['mat']['axis1'][:]]
        all_embeddings = f['mat']['block0_values'][:]
    return all_words, all_embeddings
    #len(all_words) = 362891
    #all_embeddings dimensions = (362891, 300), V x 300 matrix
    #strings are in form /c/language_code/word


#extract english words and embeddings, remove /c/en/ prefix
def english_words(all_words, all_embeddings):
    english_words = [word[6:] for word in all_words if word.startswith('/c/en/')]
    english_word_indices = [i for i, word in enumerate(all_words) if word.startswith('/c/en/')]
    english_embeddings = all_embeddings[english_word_indices]
    return english_words, english_embeddings
    #len(all_words) = 150875
    #english_embeddings dimensions = (150875, 300), V x 300 matrix

#normalize our vectors, dividing each by its length
def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1)
    normalized_embeddings = embeddings.astype('float32') / norms.astype('float32').reshape([-1, 1])
    return normalized_embeddings

def create_word_dictionary(words):
    index = {word: i for i, word in enumerate(words)}
    return index