from data_processing import load_ConceptNet, extract_embeddings, english_words, normalize_embeddings, create_word_dictionary

def load_normalized_english_embeddings():
    load_ConceptNet()
    all_words, all_embeddings = extract_embeddings()
    english_words, english_embeddings = english_words(all_words, all_embeddings)
    normalized_embeddings = normalize_embeddings(english_embeddings)
    index = create_word_dictionary(english_words)
    return english_words, normalized_embeddings, index