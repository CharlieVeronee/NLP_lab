from model_data_processing import convert_text, prepare_tensors

train_test_text = "datasets/movie-simple.txt"
def load_model_data():
    xs, ys = convert_text(train_test_text)
    train_loader, test_loader = prepare_tensors(xs,ys)
    return train_loader, test_loader