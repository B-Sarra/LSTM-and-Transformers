import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader import load_mydata, get_embed_matrix
import numpy as np
from datetime import datetime
from config import config
from LSTM import LSTMIMDB
import pandas as pd
from tqdm import tqdm
from model import Net


def accuracy_on_test_set(model, x_test, y_test, args):
    model.eval()
    with torch.no_grad():
        correct = 0.0
        total = 0.0
        num_batches = int(len(x_test) / args.batch_size)
        for j in tqdm(range(num_batches)):
            x_batch = torch.tensor(x_test[j * args.batch_size:(j + 1) * args.batch_size])
            y_batch = y_test[j * args.batch_size:(j + 1) * args.batch_size]
            curr_batch_size = x_batch.shape[0]
            hidden = (model.init_hidden(curr_batch_size)[0].to(args.device),
                      model.init_hidden(curr_batch_size)[1].to(args.device))
            model_out, _ = model(x_batch.to(args.device), hidden)
            model_out = model_out.to("cpu").numpy()
            correct += (model_out.argmax(axis=1) == y_batch).sum()
        accuracy = correct / len(x_test)
        return accuracy


# def accuracy_on_test_set(lstm_imdb,x_test, y_test, conf):
#     lstm_imdb.to('cpu')
#     right_answers = 0
#     for index, data_entry in tqdm(enumerate(x_test)):
#         data_entry = np.array(data_entry)
#         data_entry = data_entry.astype(np.int32)
#         data_entry = [[a.item() for a in data_entry]]
#         target_data = y_test[index].item()
#         # print("target_data: ", target_data)
#         input_data = autograd.Variable(torch.LongTensor(data_entry))
#         # print("input data size",input_data.size())
#         # target_data = autograd.Variable(torch.LongTensor(target_data))
#         hidden = (torch.rand(1,1,conf.hidden_dim),torch.rand(1,1,conf.hidden_dim))
#         y_pred, _ = lstm_imdb(input_data, hidden)
#         value, predicted_index = torch.max(y_pred, 1)
#         # print("value: ", value)
#         # print("predicted_index: ", predicted_index)
#         predicted_value = predicted_index.data.numpy()[0]
#         # print("predicted_value: ", predicted_value)
#         # print("target_data.data.numpy()[0]: ", target_data.data.numpy()[0])
#         # print("target_data: ", target_data)
#
#         if (predicted_value == target_data):
#             right_answers += 1
#     accuracy = right_answers / len(x_test)
#     print("Accuracy on test set: ", accuracy)
#     return accuracy

def val(model, val_x, val_y, device):
    """
        Evaluates model on the test set
    """
    model.eval()
    val_x, val_y = torch.tensor(val_x), torch.tensor(val_y)

    with torch.no_grad():
        correct = 0.0
        total = 0.0
        for i in tqdm(range(len(val_y))):
            model_out = model(val_x[i].unsqueeze(0).to(device)).to("cpu").numpy()
            correct += (model_out.argmax(axis=1) == val_y[i].numpy()).sum()
            total += 1
        accuracy = correct / total
        print("{}, {}/{}".format(accuracy, correct, total))
        return accuracy


def main():
    args = config()
    # (X_train, y_train) = load_mydata(args,'train')
    (X_train_new, y_train_new) = load_mydata(args, 'train_new')
    (X_train, y_train) = (X_train_new, y_train_new)
    (X_valid, y_valid) = load_mydata(args, 'valid')
    (X_test, y_test) = load_mydata(args, 'test')

    embeddings = torch.tensor(get_embed_matrix(args))
    X_train = torch.tensor(X_train)
    # print(X_train.size(),y_train)
    # X_train = [torch.tensor(x) for x in X_train]
    # data_length = np.asarray([len(sq) for sq in X_train])
    # X_train = torch.nn.utils.rnn.pad_sequence(X_train,batch_first=True,padding_value=0)
    # TODO: here pad with 0, different from Transformer
    num_batches = int(len(X_train) / args.batch_size)
    # restore np.load for future normal usage
    if args.model == 'bilstm':
        best_val = 0
        lstm_imdb = LSTMIMDB(embeddings, args)
        loss_function = nn.NLLLoss()
        optimizer = optim.Adam(lstm_imdb.parameters(), lr=1e-3)

        # Training epochs
        for i in range(1, args.epochs + 1):
            loss_average = 0.0
            start_time = datetime.now()
            lstm_imdb.to(args.device).train()
            shuffle_index = np.random.permutation(len(X_train))
            curr_train_text = X_train[shuffle_index]
            curr_train_label = y_train[shuffle_index]
            # curr_seq_length = data_length[shuffle_index]
            print("Training..")
            for j in tqdm(range(num_batches)):
                x_batch = curr_train_text[j * args.batch_size:(j + 1) * args.batch_size]
                y_batch = curr_train_label[j * args.batch_size:(j + 1) * args.batch_size]
                # seq_length_batch = curr_seq_length[j * FLAGS.batch_size:(j + 1)*FLAGS.batch_size]
                curr_batch_size = x_batch.size()[0]
                # TODO: user dataloader and validation and pack_padded seq
                # x_batch_pack = torch.nn.utils.rnn.pack_padded_sequence(x_batch,seq_length_batch,batch_first=True,enforce_sorted=False)#TODO: should sorted better?
                # print(x_batch_pack)
                # input_sequence = torch.tensor(x_batch,dtype=torch.long)
                # print('input size',input_sequence.size())
                y = torch.tensor(y_batch, dtype=torch.long)
                hidden = (lstm_imdb.init_hidden(curr_batch_size)[0].to(args.device),
                          lstm_imdb.init_hidden(curr_batch_size)[1].to(args.device))
                y_pred, _ = lstm_imdb(x_batch.to(args.device), hidden)
                # hidden = (lstm_imdb.init_hidden()[0],lstm_imdb.init_hidden()[1])
                # y_pred, _ = lstm_imdb(x_batch, hidden)
                # print('y pred size:',y_pred.size())
                lstm_imdb.zero_grad()
                loss = loss_function(y_pred.to('cpu'), y)
                # loss = loss_function(y_pred, y)
                # print("loss: ", loss)
                # print("loss.data[0]: ", loss.data)
                loss_average += loss.data
                # print(loss_average)
                # loss_average += loss.data[0]
                # if j % 100 == 0:
                #     # print("epoch: %d iteration: %d loss: %g" % (i, index, loss.data[0]))
                #     print("epoch: %d iteration: %d loss: %g" % (i, j, loss.data))
                loss.backward()
                optimizer.step()
            print("epoch: {} Average loss: {}".format(i, loss_average.numpy() / len(X_train)))
            time_elapsed = datetime.now() - start_time
            print("Time elapsed {}".format(time_elapsed.total_seconds()))
            # Validate on test-set every epoch

            accuracy = accuracy_on_test_set(lstm_imdb, X_valid, y_valid, args)
            print("Accuracy on validation set: ", accuracy)
            print('Best validation accuracy now:', max(accuracy, best_val), '\n')
            if accuracy > best_val:
                best_val = accuracy
                accuracy = accuracy_on_test_set(lstm_imdb, X_test, y_test, args)
                print("Accuracy on test set: ", accuracy)

        accuracy = accuracy_on_test_set(lstm_imdb, X_test, y_test, args)
        print("Accuracy on test set: ", accuracy)

    elif args.model == 'transformer':
        model = Net(
            model_size=args.hidden_dim, embeddings=embeddings,
            max_length=args.max_seq_length, num_heads=args.num_heads,
            num_blocks=args.num_blocks, dropout=args.dropout,
            train_word_embeddings=True,
        ).to(args.device)
        model = nn.DataParallel(model)

        optimizer = optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
        criterion = nn.CrossEntropyLoss()
        best_val = 0
        num_batches = int(len(X_train) / args.batch_size)
        time_cost = []
        for i in range(1, args.epochs + 1):
            loss_sum = 0.0
            start_time = datetime.now()
            model.train()
            shuffle_index = np.random.permutation(len(X_train))
            # print(shuffle_index)
            curr_train_text = X_train[shuffle_index]
            curr_train_label = y_train[shuffle_index]
            # print(curr_train_text[:10])
            print("Training..")
            for j in tqdm(range(num_batches)):
                x_batch = torch.tensor(curr_train_text[j * args.batch_size:(j + 1) * args.batch_size])
                y_batch = torch.tensor(curr_train_label[j * args.batch_size:(j + 1) * args.batch_size])
                optimizer.zero_grad()
                model_out = model(x_batch.to(args.device))
                loss = criterion(model_out, y_batch.to(args.device))
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()

            time_elapsed = datetime.now() - start_time
            time_cost.append(time_elapsed)
            print("Time elapsed {}".format(time_elapsed))
            print("Epoch: {}, Loss mean: {}".format(i, loss_sum / j))
            # Validate on test-set every epoch
            print("Validating..")
            accuracy = val(model, X_valid, y_valid, args.device)
            print('Best validation accuracy now:', max(accuracy, best_val), '\n')
            if accuracy > best_val:
                best_val = accuracy
                print("Testing..")
                val(model, X_test, y_test, args.device)
        print("average time per epoch:", np.mean(time_cost))


if __name__ == "__main__":
    main()
