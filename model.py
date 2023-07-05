import torch
nn = torch.nn

class emb_permute(nn.Module):
    def __init__(self, args):
        super(emb_permute, self).__init__()
        self.emb = nn.Embedding(args['VOCAB_SIZE'] * 2, args['EMB_DIM'])

    def forward(self, indices):
        return self.emb(indices).permute(1, 0, 2)

class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = args['DEVICE']
        self.emb = emb_permute(args=args)
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=args['EMB_DIM'],
                nhead=args['ATTN_HEAD'],
                dim_feedforward=args['FEED_DIM']
            ),
            num_layers=args['ENC_LAYERS']
        )

        self.positional = nn.Parameter(torch.randn(2 * args['SEQ_LENGTH'] + 4, 1, args['EMB_DIM']))
        # self.output_positional = nn.Parameter(torch.randn(4, 1, args['EMB_DIM']))

    # def make_target_mask(self, size):
    #     mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
    #     mask = mask.float()
    #     mask = mask.masked_fill(mask == 0, -9e3)  # Convert zeros to -inf
    #     mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0
    #     return mask.to(self.device)

    def forward(self, inputs):
        # Obtian the batch number

        B = inputs.shape[0]
        seq_length = inputs.shape[1]

        inputs = self.emb(inputs)
        inputs += self.positional[0:seq_length].repeat(1, B, 1)

        outputs = self.enc(inputs)
        return outputs # Not return the hidden states here

class RNN_encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        # self.emb = emb_permute(args=args)
        self.emb = nn.Linear(args['VOCAB_SIZE'] * 2, args['EMB_DIM'])
        if args['RNN_TYPE'] == 'LSTM':
            self.enc = nn.LSTM(
                input_size=args['EMB_DIM'],
                hidden_size=args['HIDDEN'],
                num_layers=args['LAYERS'],
                dropout=args['DROPOUT'],
                bidirectional=args['BIDIRECTION']
            )

        elif args['RNN_TYPE'] == 'GRU':
            self.enc = nn.GRU(
                input_size=args['EMB_DIM'],
                hidden_size=args['HIDDEN'],
                num_layers=args['LAYERS'],
                dropout=args['DROPOUT'],
                bidirectional=args['BIDIRECTION']
            )

        self.dropout = nn.Dropout(p=args['DROPOUT'])

    def forward(self, x):
        x = self.dropout(self.emb(x))
        x, h_c = self.enc(x)

        return x

    # def recursive(self, inputs, device, target_length=30):
    #     # Obtain the batch size
    #     B = inputs.shape[0]
    #     seq_length = inputs.shape[1]
    #
    #     inputs = self.emb(inputs) + self.positional[0:seq_length].repeat(1, B, 1)
    #     memory = self.transformer.encoder(inputs)
    #
    #     # define the outputs container shape: seq * [1, batch, feat]
    #     # outputs = []
    #
    #     # construct the decoder inputs start with '<s>'.
    #     s_tag = torch.LongTensor([26]).repeat(B, 1)
    #     s_tag = s_tag.to(device)
    #     dec_inputs = s_tag
    #     # dec_inputs = self.emb(dec_inputs) + self.output_positional[0:1].repeat(1, B, 1)
    #
    #     # Obtain the indices
    #     # output = self.hidden2tags(self.transformer.decoder(dec_inputs, memory))
    #     # outputs.append(output)
    #
    #     # Predict the remaining the sequences
    #     for i in range(target_length):
    #         # a = self.emb(dec_inputs)
    #         # b = self.output_positional[0: i+1].repeat(1, B, 1)
    #         dec_inputs_emb = self.emb(dec_inputs) + self.positional[0: i+1].repeat(1, B, 1)
    #         output_feat = self.hidden2tags(self.transformer.decoder(dec_inputs_emb, memory))
    #         outputs_indice = torch.argmax(input=output_feat, dim=-1).T
    #         dec_inputs = torch.cat([s_tag, outputs_indice], dim=1) # Concatenate the output indices to the dec_inputs
    #         # dec_inputs = torch.argmax(input=output, dim=-1).T # [seq, batch, feat] -> [seq, batch] -> [batch, seq]
    #         # a = self.output_positional[0:i + 1].repeat(1, B, 1)
    #         # dec_inputs = self.emb(dec_inputs) + self.output_positional[0:i+1].repeat(1, B, 1)
    #         # output = self.hidden2tags(self.transformer.decoder(dec_inputs, memory))
    #
    #     return output_feat

class cp_2_key_model(nn.Module):
    def __init__(self, args, out_channels, pre_trained_encoder=None):
        '''
        This model is inspired by the design from the BERT model for doing
        sentences pair classification.

        Read: https://arxiv.org/pdf/1810.04805.pdf
        :param args:
        '''
        super(cp_2_key_model, self).__init__()
        # Load the pretrained decoder of initial a new one
        self.linear_projectors = nn.ModuleList()
        if pre_trained_encoder is not None:
            self.encoder_wapper = pre_trained_encoder
            self.linear_projector = nn.Linear(args['EMB_DIM'], out_channels)

        elif args['TYPE'] == 'CP2K':
            self.encoder_wapper = Encoder(args=args)
            self.linear_projector = nn.Linear(args['EMB_DIM'], out_channels)

        elif args['TYPE'] == 'CP2K_RNN':
            self.encoder_wapper = RNN_encoder(args)
            for _ in range(3):
                self.linear_projectors.append(
                    nn.Linear(
                        args['HIDDEN'] * 2 if args['BIDIRECTION'] else args['HIDDEN'], out_channels
                    )
                )
        elif args['TYPE'] == 'CP2K_RNN_ENC':
            self.encoder_wapper = nn.Sequential(
                RNN_encoder(args),
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=args['HIDDEN'] * 2 if args['BIDIRECTION'] else args['HIDDEN'],
                        nhead=args['ATTN_HEAD'],
                        dim_feedforward=args['FEED_DIM'],
                        dropout=args['DROPOUT']
                    ),
                    num_layers=args['ENC_LAYERS']
                )
            )
            for _ in range(3):
                self.linear_projectors.append(
                    nn.Linear(
                        args['HIDDEN'] * 2 if args['BIDIRECTION'] else args['HIDDEN'], out_channels
                    )
                )
            # self.linear_projector = nn.Linear(args['HIDDEN'] * 2 if args['BIDIRECTION'] else args['HIDDEN'],
            #                                   out_channels)

    def forward(self, x):
        x = self.encoder_wapper(x) # [seq, batch, feats]
        # x = torch.mean(x, dim=0) # -> [batch, feats]
        return torch.stack([proj(x) for proj in self.linear_projectors]) # -> [3, seq, batch, out_channels]

class cp_2_k_mask(nn.Module):
    def __init__(self, args, out_channels):
        super(cp_2_k_mask, self).__init__()

        # RNN and Transformer encoder
        self.rnn = RNN_encoder(args)
        self.transformer_enc = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=args['HIDDEN'] * 2 if args['BIDIRECTION'] else args['HIDDEN'],
                        nhead=args['ATTN_HEAD'],
                        dim_feedforward=args['FEED_DIM'],
                        dropout=args['DROPOUT']
                    ),
                    num_layers=args['ENC_LAYERS']
                )

        # Fully connected layers for the final outputs
        self.linear_projectors = nn.ModuleList()
        for _ in range(3):
            self.linear_projectors.append(
                nn.Linear(
                    args['HIDDEN'] * 2 if args['BIDIRECTION'] else args['HIDDEN'], out_channels
                )
            )

    def forward(self, x, masks):
        x = self.rnn(x)
        x = self.transformer_enc(x, src_key_padding_mask=masks)
        return torch.stack([proj(x) for proj in self.linear_projectors]) # -> [3, seq, batch, out_channels]





if __name__ == "__main__":
    from config import args
    model = cp_2_key_model(args, out_channels=26)
    print(torch.randint(low=0, high=4, size=[2, 82]))
    output = model(torch.randint(low=0, high=4, size=[2, 6]))
    print(output.shape)
