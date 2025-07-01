import numpy as np
from vector_data_loader import load_normalized_english_embeddings

english_words, normalized_embeddings, index = load_normalized_english_embeddings()

#similarity between two words
def similarity_score(w1, w2):
    score = np.dot(normalized_embeddings[index[w1], :], normalized_embeddings[index[w2], :])
    return score
#max similarity word like cat and cat
#closely related word like cat and feline
#unrelated word like cat and door


#the most similar words to a given word
def closest_to_vector(v, n):
    all_scores = np.dot(normalized_embeddings, v)
    best_words = list(map(lambda i: english_words[i], reversed(np.argsort(all_scores))))
    return best_words[:n]

#prints n most similar words to w
def most_similar(w, n):
    return closest_to_vector(normalized_embeddings[index[w], :], n)

#find words "nearby" vectors that we create ourselves
#ie man:brother :: woman:? -> brother-man+woman:
def solve_analogy(a1, b1, a2):
    b2 = normalized_embeddings[index[b1], :] - normalized_embeddings[index[a1], :] + normalized_embeddings[index[a2], :]
    return closest_to_vector(b2, 1)