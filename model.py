import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.emb = nn.Embedding(args['VOCAB_SIZE'],args['EMB_DIM'])
        self.dropout = nn.Dropout(p=args['DROPOUT'])
    
        self.rnn = nn.GRU(
                    input_size=args['EMB_DIM'],
                    hidden_size=args['HIDDEN'],
                    dropout=args['DROPOUT'],
                    bidirectional=args['BIDIRECTION'],
                    num_layers=args['LAYERS'],
                ) if args['RNN_TYPE'] == 'GRU' else nn.LSTM(
                    input_size=args['EMB_DIM'],
                    hidden_size=args['HIDDEN'],
                    dropout=args['DROPOUT'],
                    bidirectional=args['BIDIRECTION'],
                    num_layers=args['LAYERS'],
                )

        self.decode = nn.Linear(args['HIDDEN'] * 2 if args['BIDIRECTION'] else args['HIDDEN'], args['VOCAB_SIZE'])


    def forward(self, x):
        '''


        :param x: The indice input with shape [batch, seq]
        :return: plaintext or deciphered output.
        '''
        x = self.dropout(self.emb(x)).permute(1, 0, 2) # [batch, seq] -> [seq, batch, embedding]
        output = self.rnn(x) # Obtain the outputs and hidden states or cell

        return self.decode(output[0]) # Not return the hidden states here



