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
        # self.emb = nn.Linear(args['VOCAB_SIZE'] * 2, args['EMB_DIM'])
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

    def forward(self, x, **kwargs):
        # x = self.dropout(self.emb(x))
        x, h_c = self.enc(x)
        return x



class cp_2_k_mask(nn.Module):
    def __init__(self, args, out_channels):
        super(cp_2_k_mask, self).__init__()
        self.out_channels = out_channels
        self.rotor_num = len(args['ROTOR'].split(' '))

        self.networks = nn.ModuleList()
        self.linear_projectors = nn.ModuleList()

        # Embedding (zeros if there exists LSTM layers)
        if args['LAYERS'] == 0:
            self.position_emb = nn.Parameter(torch.randn(60, 1, args['EMB_DIM'] * 2)).to(args['DEVICE'])
        else:
            self.position_emb = torch.zeros(60, 1, args['EMB_DIM']).to(args['DEVICE'])

        self.emb = nn.Linear(args['VOCAB_SIZE'] * 2, args['EMB_DIM'] if args['LAYERS'] > 0 else args['EMB_DIM'] * 2)

        # Dropout
        self.dropout = nn.Dropout(p=args['DROPOUT'])

        # Adding LSTM layers
        if args['LAYERS'] > 0:
            self.networks.append(RNN_encoder(args))

        # Adding Transformer layers
        if args['ENC_LAYERS'] > 0:
            self.networks.append(nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=args['HIDDEN'] * 2,
                        nhead=args['ATTN_HEAD'],
                        dim_feedforward=args['FEED_DIM'],
                        dropout=args['DROPOUT']
                    ),
                    num_layers=args['ENC_LAYERS']
                ))

        # linear projectors for predictions
        self.linear_projectors = nn.Linear(args['HIDDEN'] * 2, out_channels * 3)

        # Further experiments projector
        self.num_rand_rotor_type = args['Random_rotors_num']
        self.random_reflector =  args['Random_reflector']
        self.num_rand_ringsetting = args['Random_ringsetting_num']
        self.num_rand_plug = args['Random_plugboard_num']



        # self.further_experiment_proj = nn.ModuleList()
        # random rotors (8 different types of rotors) shape : [1 - 3, 8]
        if args['Random_rotors_num'] is not None:
            # shape [batch, feats] -> [batch, rotor_num, rotor_type(8)]
            self.rotor_type_proj = nn.Linear(args['HIDDEN'] * 2, self.num_rand_rotor_type * 8)

        # random reflector (B and C) shape : [1]
        if args['Random_reflector'] is not None:
            # Shape: [batch, feats] -> [batch]
            self.reflector_proj = nn.Linear(args['HIDDEN'] * 2, 2)

        # random ring setting (26 ringsetting for each rotor) shape : [1 - 3, 26]
        if args['Random_ringsetting_num'] is not None:
            # shape: [batch, feats] -> [batch, ringsetting_num, 26]
            self.ringsetting_proj = nn.Linear(args['HIDDEN'] * 2, self.num_rand_ringsetting * 26)

        # Random plugboard( 26x 26 plugboard setting) shape : [26, 26]
        if args['Random_plugboard_num'] is not None:
            # shape: [batch, feats] -> [batch, 26, 26]
            self.plugboard_proj = nn.Linear(args['HIDDEN'] * 2, 26 * 26)


    def forward(self, x, masks):
        # Embedding
        x = self.dropout(self.emb(x))

        # Obtain the length of dimension
        seq, batch, features = x.shape

        # Adding the position embedding
        x = x + self.position_emb[:seq].repeat(1, batch, 1)

        # forwarding
        for layer in self.networks:
            x = layer(x, src_key_padding_mask=masks)

        # predictions
        position_pred = self.linear_projectors(x).reshape(seq, batch, 3, self.out_channels).permute(2, 0, 1, 3)

        # Further prediction
        # From here we place the features on the second dimension because the Cross Entropy loss works on the second dim
        further = []
        x = torch.mean(x, dim=0)
        if self.num_rand_rotor_type is not None:
            further.append(self.rotor_type_proj(x).reshape(batch, 8, self.num_rand_rotor_type))

        if self.random_reflector is not None:
            further.append(self.reflector_proj(x))

        if self.num_rand_ringsetting is not None:
            further.append(self.ringsetting_proj(x).reshape(batch, 26, self.num_rand_ringsetting))

        if self.num_rand_plug is not None:
            further.append(self.plugboard_proj(x).reshape(batch, 26, 26))


        return position_pred, further
        # return torch.stack([proj(x) for proj in self.linear_projectors])  # -> [3, seq, batch, out_channels]


class cp_2_k_onnx(cp_2_k_mask):
    def __init__(self, args, out_channels):
        super().__init__(args, out_channels)

    def forward(self, x):
        # Embedding
        x = self.dropout(self.emb(x))

        seq, batch, features = x.shape

        # forwarding
        for layer in self.networks:
            x = layer(x)

        # predictions
        return self.linear_projectors(x).view(seq, batch, 3, self.out_channels).permute(2, 0, 1, 3)



if __name__ == "__main__":
    from config import args
    sample = torch.randn(args['SEQ_LENGTH'][1], 1, args['VOCAB_SIZE'] * 2)
    print(sample.shape)
    print(sample.repeat(1, 5, 1).shape)
