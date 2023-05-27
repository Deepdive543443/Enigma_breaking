import torch
nn = torch.nn

class RNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.emb = nn.Embedding(args['VOCAB_SIZE'],args['EMB_DIM'])
        self.dropout = nn.Dropout(p=args['DROPOUT'])
        if args['RNN_TYPE'] == 'GRU':
    
            self.rnn = nn.GRU(
                        input_size=args['EMB_DIM'],
                        hidden_size=args['HIDDEN'],
                        dropout=args['DROPOUT'],
                        bidirectional=args['BIDIRECTION'],
                        num_layers=args['LAYERS'],
                    )

        elif args['RNN_TYPE'] == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=args['EMB_DIM'],
                hidden_size=args['HIDDEN'],
                dropout=args['DROPOUT'],
                bidirectional=args['BIDIRECTION'],
                num_layers=args['LAYERS'],
            )
        #
        # elif args['RNN_TYPE'] == 'TRANSFORMER':
        #     encoder_layer = nn.TransformerEncoderLayer(d_model=args['HIDDEN'], nhead=args['ATTN_HEAD'])
        #     self.rnn = nn.TransformerEncoder(
        #         encoder_layer=encoder_layer, num_layers=args['LAYERS']
        #     )

        self.seq_decode = nn.Linear(args['HIDDEN'] * 2 if args['BIDIRECTION'] else args['HIDDEN'], args['VOCAB_SIZE'])
        # self.key_decode = nn.LSTM(
        #     input_size=args['HIDDEN'] * 2,
        #     hidden_size=args['HIDDEN'],
        #     dropout=args['DROPOUT'],
        #     num_layers=args['LAYERS'] * 2,
        # )

    def forward(self, x):
        '''


        :param x: The indice input with shape [batch, seq]
        :return: plaintext or deciphered output.
        '''
        x = self.dropout(self.emb(x)).permute(1, 0, 2) # [batch, seq] -> [seq, batch, embedding]
        output = self.rnn(x)

        return self.seq_decode(output[0]) # Not return the hidden states here


class TransformerPart(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.emb = nn.Embedding(args['VOCAB_SIZE'], args['EMB_DIM'])
        encoder_layer = nn.TransformerEncoderLayer(d_model=args['EMB_DIM'], nhead=args['ATTN_HEAD'])
        self.tfm = nn.TransformerEncoder(
                encoder_layer=encoder_layer, num_layers=args['LAYERS']
            )

        self.positional = nn.Parameter(torch.randn(args['SEQ_LENGTH'] + 3, 1, args['EMB_DIM']))

    def forward(self, x):
        '''


        :param x: The indice input with shape [batch, seq]
        :return: plaintext or deciphered output.
        '''
        x = self.dropout(self.emb(x)).permute(1, 0, 2) # [batch, seq] -> [seq, batch, embedding]
        output = self.rnn(x)
        return self.decode(output[0]) # Not return the hidden states here
#


if __name__ == "__main__":
    from config import args
    import torch.optim as optim
    model = RNN(args=args)
    model.state_dict()

    adam = optim.Adam(model.parameters(), lr=3e-4)
    print(adam.state_dict())

