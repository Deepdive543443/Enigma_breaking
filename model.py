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
        output = self.rnn(x)
        return self.decode(output[0]) # Not return the hidden states here

if __name__ == "__main__":
    from config import args
    import torch.optim as optim
    model = RNN(args=args)
    model.state_dict()

    adam = optim.Adam(model.parameters(), lr=3e-4)
    print(adam.state_dict())

