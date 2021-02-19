from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe
from config import config
import numpy as np
import pandas as pd
import string
import torch
import math
import json
import os
import pickle
import gluonnlp
import spacy
# nlp = spacy.load("en")

def get_token_id(data,vocab):
    tokens = []
    for sen in data:
        token = [word.text for word in nlp(sen)]
        # mean is 273.564
        token2idx = vocab[token[:args.max_seq_length]]
        while len(token2idx) < args.max_seq_length:
            token2idx.append(vocab['<pad>'])
        tokens.append(token2idx)
    return np.asarray(tokens)

def corpus_process():
    # TODO: try to repalce it with torchtext vocab?
    print(gluonnlp.embedding.list_sources('glove'))
    glove = gluonnlp .embedding.create('glove', source='glove.6B.50d')
    vocab = gluonnlp .Vocab(gluonnlp.data.Counter(glove.idx_to_token))
    vocab.set_embedding(glove)
    # print(vocab['<pad>','<unk>'])
    # print(vocab.idx_to_token[3])
    embeddings = vocab.embedding.idx_to_vec
    # We use imdb5k first
    data_train = pd.read_csv(os.path.join(args.data_path, 'imdb5k_train.csv'))
    data_test = pd.read_csv(os.path.join(args.data_path, 'imdb5k_test.csv'))
    data_train.replace(to_replace='neg', value=0, inplace=True)
    data_train.replace(to_replace='pos', value=1, inplace=True)
    data_test.replace(to_replace='neg', value=0, inplace=True)
    data_test.replace(to_replace='pos', value=1, inplace=True)
    X_train, y_train = get_token_id(data_train['text'],vocab) ,np.asarray(data_train['label'])
    X_test, y_test = get_token_id(data_test['text'],vocab), np.asarray(data_test['label'])
    # print(len(y_train))
    X_train_new,y_train_new = X_train[:4000],y_train[:4000]
    X_valid, y_valid = X_train[4000:],y_train[4000:]
    # print(X_train_new,len(y_train_new),len(y_valid))
    train = (X_train,y_train)
    train_new = (X_train_new,y_train_new)
    valid = (X_valid, y_valid)
    test = (X_test,y_test)
    pickle.dump(train, open(args.data_path + '/train.pkl', 'wb'))
    pickle.dump(train_new, open(args.data_path + '/train_new.pkl', 'wb'))
    pickle.dump(valid, open(args.data_path + '/valid.pkl', 'wb'))
    pickle.dump(test, open(args.data_path + '/test.pkl', 'wb'))
    pickle.dump(embeddings, open(args.data_path + '/embedding_matrix', 'wb'))


def load_mydata(args,type):
    data = None
    if type == 'train':
        print('Getting training data...')
        with open(args.data_path+'/train.pkl',  'rb') as f:
            print(f.name)
            data = pickle.load(f)
    elif type == 'valid':
        print('Getting validating new data...')
        with open(args.data_path + '/valid.pkl', 'rb') as f:
            print(f.name)
            data = pickle.load(f)
    elif type == 'train_new':
        print('Getting training new data...')
        with open(args.data_path + '/train_new.pkl', 'rb') as f:
            print(f.name)
            data = pickle.load(f)
    elif type == 'test':
        print('Getting test data...')
        with open(args.data_path + '/test.pkl', 'rb') as f:
            print(f.name)
            data = pickle.load(f)
    assert data
    return data

def generate_batch(args, data):
    num_batch = int(math.ceil(len(data)/args.batch_size))
    for i in range(num_batch):
        batch = data[i*args.batch_size:(i+1)*args.batch_size]
        # if len(batch) == args.batch_size: #TODO: last batch not used!
        yield batch

def get_embed_matrix(args):
    print('Getting embedding_matrix')
    with open(args.data_path + '/embedding_matrix', 'rb') as f:
        return pickle.load(f).asnumpy()


def create_feed_dict(batch):
    """
    :param is_train: Flag, True for train batch
    :param batch: list train/evaluate data
    :return: structured data to feed
    """
    token_idx, target_idx, target_loc, labels = [], [], [], []
    for line in batch:
        x = line[0]
        labels.append(line[1])
        token_idx.append(x[0])
        target_idx.append(x[1])
        target_loc.append(x[2])
    feed_dict = {
            'tokens': np.asarray(token_idx),
            'target': np.asarray(target_idx),
            'loc': np.asarray(target_loc),
            'labels':np.asarray(labels)
        }
    return feed_dict

def create_test_feed(line):
    """
    :param is_train: Flag, True for train batch
    :param batch: list train/evaluate data
    :return: structured data to feed
    """
    token_idx, target_idx, target_loc, labels = [], [], [], []



    feed_dict = {
            'tokens': np.asarray(token_idx),
            'target': np.asarray(target_idx),
            'loc': np.asarray(target_loc),
            'labels':np.asarray(labels)
        }
    return feed_dict


def tokenize(input):
    """
        Naive tokenizer, that lower-cases the input
        and splits on punctuation and whitespace
    """
    input = input.lower()
    for p in string.punctuation:
        input = input.replace(p," ")
    return input.strip().split()


def num2words(vocab,vec):
    """
        Converts a vector of word indicies
        to a list of strings
    """
    return [vocab.itos[i] for i in vec]




def get_input_tensor(vocab, data, max_length):
    out = {}
    text_index_all = []
    label_all = []
    for sample in data:
        text_index = [vocab.stoi[i] for i in sample.text][:max_length]
        while len(text_index)< max_length:
            text_index.append(1)
        text_index_all.append(text_index)
        label = 1 if sample.label == 'pos' else 0
        label_all.append(label)

    text_index_all = torch.tensor(text_index_all,dtype=torch.long)
    label_all = torch.tensor(label_all, dtype=torch.long)
    # print(text_index_all.size(), label_all.size())
    out['text']=text_index_all
    out['label'] = label_all
    return out



def get_imdb(batch_size,max_length):
    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True,tokenize=tokenize,fix_length=max_length)
    LABEL = data.Field(sequential=False,unk_token=None,pad_token=None)

    print("Loading data..\n")

    # make splits for data
    train, test = datasets.IMDB.splits(TEXT, LABEL)

    # print information about the data
    print('train.fields', train.fields)
    print('length of all train data', len(train))
    print('length of test data', len(test))
    print("")

    # build the vocabulary
    TEXT.build_vocab(train, vectors=GloVe(name='42B', dim=300, max_vectors=500000))
    LABEL.build_vocab(train)

    # print vocab information
    print('length of TEXT.vocab', len(TEXT.vocab))
    print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())

    # make iterator for splits

    train_iter, test_iter = data.BucketIterator.splits(
        (train,test), batch_size=batch_size)

    # split validation data
    train_new = list(train)[:20000]
    validation = list(train)[20000:]
    train_new =  get_input_tensor(TEXT.vocab,train_new,max_length)
    validation = get_input_tensor(TEXT.vocab,validation,max_length)

    # train,test = list(train),list(test)
    return train_iter, train_new,validation,test_iter, TEXT.vocab.vectors, TEXT.vocab

if __name__ == "__main__":
    """
        If run seperately, does a simple sanity check,
        by printing different values,
        and counting labels
    """
    args = config()
    corpus_process()


