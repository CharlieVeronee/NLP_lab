from vector_data_processing import load_ConceptNet, extract_embeddings, english_words, normalize_embeddings, create_word_dictionary

def load_normalized_english_embeddings():
    load_ConceptNet()
    all_words, all_embeddings = extract_embeddings()
    eng_words, eng_embeddings = english_words(all_words, all_embeddings)
    norm_embeddings = normalize_embeddings(eng_embeddings)
    index = create_word_dictionary(eng_words)
    return eng_words, norm_embeddings, index