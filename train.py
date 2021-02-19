import torch
from torch import optim
from torch import nn
from dataloader import get_imdb
from model import Net
import argparse
from tqdm import tqdm
import numpy as np
from datetime import datetime

# try:
#     # try to import tqdm for progress updates
#     from tqdm import tqdm
# except ImportError:
#     # on failure, make tqdm a noop
#     def tqdm(x):
#         return x

# try:
#     # try to import visdom for visualisation of attention weights
#     import visdom
#     from helpers import plot_weights
#     vis = visdom.Visdom()
# except ImportError:
#     vis = None
#     pass

def val(model,validation,device):
    """
        Evaluates model on the test set
    """
    val_x = validation['text']
    val_y = validation['label']
    model.eval()
    print("Validating..")
    with torch.no_grad():
        correct = 0.0
        total = 0.0
        for i in tqdm(range(len(val_y))):
            model_out = model(val_x[i].unsqueeze(0).to(device)).to("cpu").numpy()
            correct += (model_out.argmax(axis=1) == val_y[i].numpy()).sum()
            total += 1
        accuracy = correct / total
        print("{}, {}/{}".format(accuracy,correct,total))
        return accuracy

def test(model,test,vocab,device):
    """
        Evaluates model on the test set
    """
    model.eval()

    print("Testing..")

    # if not vis is None:
    #     visdom_windows = None
    with torch.no_grad():
        correct = 0.0
        total = 0.0
        for i,b in enumerate(tqdm(test)):
            # if not vis is None and i == 0:
            #     visdom_windows = plot_weights(model,visdom_windows,b,vocab,vis)

            model_out = model(b.text[0].to(device)).to("cpu").numpy()
            correct += (model_out.argmax(axis=1) == b.label.numpy()).sum()
            total += b.label.size(0)
        print("{}, {}/{}\n".format(correct / total,correct,total))

def train(max_length,model_size,
            epochs,learning_rate,
            device,num_heads,num_blocks,
            dropout,train_word_embeddings,
            batch_size):
    """
        Trains the classifier on the IMDB sentiment dataset
    """
    train_iter, train_new,validation, test_iter, vectors, vocab = get_imdb(batch_size = batch_size,max_length=max_length)

    # TODO: compare training time for two models

    model = Net(
                model_size=model_size,embeddings=vectors,
                max_length=max_length,num_heads=num_heads,
                num_blocks=num_blocks, dropout=dropout,
                train_word_embeddings=train_word_embeddings,
                ).to(device)
    model = nn.DataParallel(model)

    optimizer = optim.Adam((p for p in model.parameters() if p.requires_grad),lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val = 0
    num_batches = int(len(train_new['text'])/batch_size)
    time_cost = []

    for i in range(1,epochs+1):
        loss_sum = 0.0
        start_time = datetime.now()
        model.train()
        shuffle_index = np.random.permutation(len(train_new['text']))
        # print(shuffle_index)
        curr_train_text = train_new['text'][shuffle_index]
        curr_train_label = train_new['label'][shuffle_index]
        print(curr_train_text[:10])
        print("Training..")
        for j in tqdm(range(num_batches)):
            x_batch = curr_train_text[j*batch_size:(j+1)*batch_size]
            y_batch = curr_train_label[j * batch_size:(j + 1)*batch_size]
            print(x_batch.size())
            optimizer.zero_grad()
            model_out = model(x_batch.to(device))
            loss = criterion(model_out, y_batch.to(device))
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        # Retrain the model on all training dataset
        # for j, b in enumerate(iter(tqdm(train_iter))): #Done: iterator will auto make itTODO:how to shuffle the iterator
        #     optimizer.zero_grad()
        #     model_out = model(b.text[0].to(device))
        #     loss = criterion(model_out, b.label.to(device))  #TODO: add weight decay
        #     loss.backward()
        #     optimizer.step()
        #     loss_sum += loss.item()
        #
        # time_elapsed = datetime.now() - start_time
        # time_cost.append(time_elapsed)
        # print("Time elapsed {}".format(time_elapsed))
        # print("Epoch: {}, Loss mean: {}".format(i,loss_sum / j))
        # Validate on test-set every epoch
        # accuracy = val(model,validation,device)
        # print('Best validation accuracy now:',max(accuracy,best_val),'\n')
        # if accuracy>best_val:
        #     best_val = accuracy
    print("average time per epoch:",np.mean(time_cost))
    test(model, test_iter, vocab, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Transformer network for sentiment analysis")

    parser.add_argument("--max_length", default=500, type=int, help="Maximum sequence length, \
                                                                     sequences longer than this are truncated")

    parser.add_argument("--model_size", default=128, type=int, help="Hidden size for all \
                                                                     hidden layers of the model")

    parser.add_argument("--epochs", default=8, type=int, help="Number of epochs to train for")

    parser.add_argument("--learning_rate", default=0.001, type=float, dest="learning_rate",
                        help="Learning rate for optimizer")

    parser.add_argument("--device", default="cuda:0", dest="device", help="Device to use for training \
                                                                     and evaluation e.g. (cpu, cuda:0)")
    parser.add_argument("--num_heads", default=8, type=int, dest="num_heads", help="Number of attention heads in \
                                                                     the Transformer network")

    parser.add_argument("--num_blocks", default=4, type=int, dest="num_blocks",
                        help="Number of blocks in the Transformer network")

    parser.add_argument("--dropout", default=0.6, type=float, dest="dropout", help="Dropout (not keep_prob, but probability of ZEROING \
                                                                     during training, i.e. keep_prob = 1 - dropout)")

    parser.add_argument("--train_word_embeddings", type=bool, default=True, dest="train_word_embeddings",
                        help="Train GloVE word embeddings")

    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    args = vars(parser.parse_args())



    train(**args)



