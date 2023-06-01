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
class emb_permute(nn.Module):
    def __init__(self, args):
        super(emb_permute, self).__init__()
        self.emb = nn.Embedding(args['VOCAB_SIZE'], args['EMB_DIM'])

    def forward(self, indices):
        '''
        Permute the dimensions to [seq, batch, features] by default

        :param indices: input indices with shape of [batch, indices]. Type: torch.LongTensor
        :return: permuted tensor in shape [seq, batch, features]
        '''
        return self.emb(indices).permute(1, 0, 2)

class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.emb = emb_permute(args=args)
        self.transformer = nn.Transformer(
            d_model=args['EMB_DIM'], 
            nhead=args['ATTN_HEAD'],
            num_encoder_layers=args['LAYERS'],
            num_decoder_layers=args['LAYERS'],
            dim_feedforward=args['HIDDEN'],
            dropout=args['DROPOUT'], 

        )

        self.hidden2tags = nn.Linear(args['EMB_DIM'], args['VOCAB_SIZE'])
        self.positional = nn.Parameter(torch.randn(2 * args['SEQ_LENGTH'] + 4, 1, args['EMB_DIM']))
        # self.output_positional = nn.Parameter(torch.randn(4, 1, args['EMB_DIM']))

    def forward(self, inputs, targets):
        # Obtian the batch number

        B = inputs.shape[0]
        seq_length = inputs.shape[1]
        targets_length = targets.shape[1] - 1

        inputs = self.emb(inputs)
        targets = self.emb(targets[:, :-1])

        
        inputs += self.positional[0:seq_length].repeat(1, B, 1)
        targets += self.positional[0:targets_length].repeat(1, B, 1)
        
        outputs = self.transformer(
            inputs,
            targets
        )

        outputs = self.hidden2tags(outputs)
        return outputs # Not return the hidden states here

    def recursive(self, inputs, device, target_length=30):
        # Obtain the batch size 
        B = inputs.shape[0]
        seq_length = inputs.shape[1]

        inputs = self.emb(inputs) + self.positional[0:seq_length].repeat(1, B, 1)
        memory = self.transformer.encoder(inputs)

        # define the outputs container shape: seq * [1, batch, feat]
        # outputs = []

        # construct the decoder inputs start with '<s>'.
        s_tag = torch.LongTensor([26]).repeat(B, 1)
        s_tag = s_tag.to(device)
        dec_inputs = s_tag
        # dec_inputs = self.emb(dec_inputs) + self.output_positional[0:1].repeat(1, B, 1)

        # Obtain the indices
        # output = self.hidden2tags(self.transformer.decoder(dec_inputs, memory))
        # outputs.append(output)

        # Predict the remaining the sequences
        for i in range(target_length):
            # a = self.emb(dec_inputs)
            # b = self.output_positional[0: i+1].repeat(1, B, 1)
            dec_inputs_emb = self.emb(dec_inputs) + self.positional[0: i+1].repeat(1, B, 1)
            output_feat = self.hidden2tags(self.transformer.decoder(dec_inputs_emb, memory))
            outputs_indice = torch.argmax(input=output_feat, dim=-1).T
            dec_inputs = torch.cat([s_tag, outputs_indice], dim=1) # Concatenate the output indices to the dec_inputs
            # dec_inputs = torch.argmax(input=output, dim=-1).T # [seq, batch, feat] -> [seq, batch] -> [batch, seq]
            # a = self.output_positional[0:i + 1].repeat(1, B, 1)
            # dec_inputs = self.emb(dec_inputs) + self.output_positional[0:i+1].repeat(1, B, 1)
            # output = self.hidden2tags(self.transformer.decoder(dec_inputs, memory))

        return output_feat




if __name__ == "__main__":
    from config import args
    import torch.optim as optim
    model = Transformer(args=args)

    adam = optim.Adam(model.parameters(), lr=3e-4)
    print(adam.state_dict()['param_groups'][0]['lr'])

