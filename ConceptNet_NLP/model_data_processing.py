import string
import numpy as np
import torch
from vector_data_loader import load_normalized_english_embeddings

english_words, normalized_embeddings, index = load_normalized_english_embeddings()

remove_punct=str.maketrans('','',string.punctuation)

#converts a line of our data file into
#a tuple (x, y), where x is 300-dimensional representation of the words in a review, and y is its label (0 or 1)
def convert_line_to_vector(line):
    #pull out the first character: that's our label (0 or 1)
    y = int(line[0])
    
    #split the line into words using Python's split() function
    words = line[2:].translate(remove_punct).lower().split()
    
    #look up the embeddings of each word, ignoring words not in our pretrained vocabulary
    embeddings = [normalized_embeddings[index[w]] for w in words
                  if w in index]
    
    #take the mean of the embeddings
    x = np.mean(np.vstack(embeddings), axis=0)
    return x, y


#apply the function to each line in the file
def convert_text(text_url):
    xs = []
    ys = []
    with open(text_url, "r", encoding='utf-8', errors='ignore') as f:
        for l in f.readlines():
            x, y = convert_line_to_vector(l)
            xs.append(x)
            ys.append(y)

    #concatenate all examples into a numpy array
    xs = np.vstack(xs) #array of vectors (averaged per review)
    ys = np.vstack(ys) #array of labels (0 or 1)
    return xs, ys

#shuffle data and convert to tensors
#train on 80%, test on 20%
#create a TensorDataset and DataLoader for batching
def prepare_tensors(xs,ys):
    num_examples = xs.shape[0]
    shuffle_idx = np.random.permutation(num_examples)
    xs = xs[shuffle_idx, :]
    ys = ys[shuffle_idx, :]
    num_train = 4*num_examples // 5

    x_train = torch.tensor(xs[:num_train], dtype=torch.float32)
    y_train = torch.tensor(ys[:num_train], dtype=torch.float32)

    x_test = torch.tensor(xs[num_train:], dtype=torch.float32)
    y_test = torch.tensor(ys[num_train:], dtype=torch.float32)

    reviews_train = torch.utils.data.TensorDataset(x_train, y_train)
    reviews_test = torch.utils.data.TensorDataset(x_test, y_test)

    train_loader = torch.utils.data.DataLoader(reviews_train, batch_size=100, shuffle=True)
    test_loader = torch.utils.data.DataLoader(reviews_test, batch_size=100, shuffle=False)

    return train_loader, test_loader
