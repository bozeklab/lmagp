import pickle

def unpickle_dict_and_get_value(fn, mode='rb', key='features'):
    with open(fn, mode) as fp:
        return pickle.load(fp)[key]