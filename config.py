import argparse
import os

def config():
    parser = argparse.ArgumentParser()
    working_path = os.getcwd()
    data_path = os.path.join(working_path,'data')
    middle_path = os.path.join(working_path,'data')
    out_path = os.path.join(working_path,'output')

    ## Required parameters

    parser.add_argument("--model", default='bilstm', type=str,
                        help="model selected in the list: bilstm, transformer, gcn, gat")
    parser.add_argument("--task", default='w_only', type=str,
                        help="tasks: writer only; all sources")

    parser.add_argument("--working_path", default=working_path, type=str,
                        help="current working path")
    parser.add_argument("--data_path", default= data_path,type=str,
                        help="current data path")
    parser.add_argument("--data_version", default='mpqa3_1118.json', type=str,
                        help="current data version")
    parser.add_argument("--pretrained_file", default='glove.6B.50d', type=str,
                        help="Choose pretrained word embedding sources: "
                             "'glove.42B.300d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d', "
                             "'glove.6B.50d', 'glove.840B.300d', 'glove.twitter.27B.100d', 'glove.twitter.27B.200d', "
                             "'glove.twitter.27B.25d', 'glove.twitter.27B.50d'")
    parser.add_argument("--embed_dim", default=50, type=int,
                        help="word embedding dimension")
    parser.add_argument("--target_loc_dim", default=50, type=int,
                        help="target location embedding dimension")
    parser.add_argument("--max_seq_length", default=300, type=int,
                        help="Average: 31.9. The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")

    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", default=4, type=int, help="Number of epochs to train for")

    parser.add_argument("--learning_rate", default=0.001, type=float, dest="learning_rate",
                        help="Learning rate for optimizer")
    parser.add_argument("--device", default="cpu", dest="device", help="Device to use for training \
                                                                           and evaluation e.g. (cpu, cuda:0)")

    # Model setting

    parser.add_argument("--hidden_dim", default=128, type=int, help="Hidden size for all \
                                                                        hidden layers of the model")

    parser.add_argument("--num_heads", default=8, type=int, dest="num_heads", help="Number of attention heads in \
                                                                        the Transformer network")

    parser.add_argument("--num_blocks", default=4, type=int, dest="num_blocks",
                        help="Number of blocks in the Transformer network")

    parser.add_argument("--dropout", default=0.6, type=float, dest="dropout", help="Dropout (not keep_prob, but probability of ZEROING \
                                                                        during training, i.e. keep_prob = 1 - dropout)")

    parser.add_argument("--middle_path", default=middle_path, type=str,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--out_path", default=out_path, type=str,
                        help="The output directory where the model checkpoints and predictions will be written.")

    parser.add_argument("--dropout1", type=float, default=0.1, help="Dropout probability before LSTM layer")
    parser.add_argument("--dropout2", type=float, default=0.1, help="Dropout probability after LSTM layer")
    parser.add_argument("--dropout_rnn", type=float, default=0.2, help="Dropout probability within LSTM layer")
    args = parser.parse_args()
    # print(args)
    return args

