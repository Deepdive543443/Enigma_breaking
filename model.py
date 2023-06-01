import torch
nn = torch.nn

# class RNN(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.emb = nn.Embedding(args['VOCAB_SIZE'],args['EMB_DIM'])
#         self.dropout = nn.Dropout(p=args['DROPOUT'])
#         if args['RNN_TYPE'] == 'GRU':
    
#             self.rnn = nn.GRU(
#                         input_size=args['EMB_DIM'],
#                         hidden_size=args['HIDDEN'],
#                         dropout=args['DROPOUT'],
#                         bidirectional=args['BIDIRECTION'],
#                         num_layers=args['LAYERS'],
#                     )

#         elif args['RNN_TYPE'] == 'LSTM':
#             self.rnn = nn.LSTM(
#                 input_size=args['EMB_DIM'],
#                 hidden_size=args['HIDDEN'],
#                 dropout=args['DROPOUT'],
#                 bidirectional=args['BIDIRECTION'],
#                 num_layers=args['LAYERS'],
#             )
#         #
#         # elif args['RNN_TYPE'] == 'TRANSFORMER':
#         #     encoder_layer = nn.TransformerEncoderLayer(d_model=args['HIDDEN'], nhead=args['ATTN_HEAD'])
#         #     self.rnn = nn.TransformerEncoder(
#         #         encoder_layer=encoder_layer, num_layers=args['LAYERS']
#         #     )

#         self.seq_decode = nn.Linear(args['HIDDEN'] * 2 if args['BIDIRECTION'] else args['HIDDEN'], args['VOCAB_SIZE'])
#         # self.key_decode = nn.LSTM(
#         #     input_size=args['HIDDEN'] * 2,
#         #     hidden_size=args['HIDDEN'],
#         #     dropout=args['DROPOUT'],
#         #     num_layers=args['LAYERS'] * 2,
#         # )

#     def forward(self, x):
#         '''


#         :param x: The indice input with shape [batch, seq]
#         :return: plaintext or deciphered output.
#         '''
#         x = self.dropout(self.emb(x)).permute(1, 0, 2) # [batch, seq] -> [seq, batch, embedding]
#         output = self.rnn(x)

#         return self.seq_decode(output[0]) # Not return the hidden states here


class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.emb = nn.Embedding(args['VOCAB_SIZE'], args['EMB_DIM'])
        self.transformer = nn.Transformer(
            d_model=args['EMB_DIM'], 
            nhead=args['ATTN_HEAD'],
            dropout=args['DROPOUT'], 

        )

        self.hidden2tags = nn.Linear(args['EMB_DIM'], args['VOCAB_SIZE'])
        self.positional = nn.Parameter(torch.randn(2 * args['SEQ_LENGTH'] + 4, 1, args['EMB_DIM']))
        self.output_positional = nn.Parameter(torch.randn(4, 1, args['EMB_DIM']))

    def forward(self, inputs, targets):
        # Obtian the batch number

        B = inputs.shape[0]
        inputs = self.emb(inputs).permute(1, 0, 2)
        targets = self.emb(targets[:, :-1]).permute(1, 0, 2)
        
        inputs += self.positional.repeat(1, B, 1)
        targets += self.output_positional.repeat(1, B, 1)
        
        outputs = self.transformer(
            inputs,
            targets
        )

        outputs = self.hidden2tags(outputs)
        return outputs # Not return the hidden states here

    def recursive(self, inputs, device):
        # Obtain the batch size 
        B = inputs.shape[0]
        seq_length = inputs.shape[1]

        inputs = inputs + self.positional[0:seq_length].repeat(1, B, 1)
        memory = self.transformer.encoder(inputs)

        # Send in the start tag '<s>'.
        start_tag = torch.LongTensor(26).repeat(B, 1)
        start_tag = start_tag.to(device)

        start_tag = self.emb(start_tag) + self.output_positional[0:1].repeat(1, B, 1)

        outputs = []

        outputs.append(
           self.transformer.decoder(start_tag, memory) 
        )

        # Make the 
        for i in range(1, 4):
            pass


if __name__ == "__main__":
    from config import args
    import torch.optim as optim
    model = Transformer(args=args)
    model.state_dict()

    adam = optim.Adam(model.parameters(), lr=3e-4)
    print(adam.state_dict())

