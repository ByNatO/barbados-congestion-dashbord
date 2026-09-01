import pickle

def load_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model
