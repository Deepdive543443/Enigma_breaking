import os
from time import gmtime, strftime

args = {
    # Training
    'DEVICE': 'cuda',
    'BATCH_SIZE' : 512,
    'LEARNING_RATE' : 3e-4,  #512 ** (-0.5),# ,
    'BETA1': 0.9,
    'BETA2': 0.98,
    'EPS': 1e-9,
    'EPOCHS' : 1000,
    'WARMUP_STEP': 2000,

    # Model configuraion
    'TYPE': 'Encoder', # 'Encoder' for sequence labeling Or 'CP2K' for text pair classification
    'LAYERS' : 6,
    'EMB_DIM' : 512,
    'HIDDEN': 2048,
    'DROPOUT': 0.1,

    # Pretrained setting
    'PRE_TRAINED_ENC': None,

    # Transformer configuration
    'ATTN_HEAD': 8,
    # RNN Configuration
    'RNN_TYPE' : 'LSTM',
    'BIDIRECTION' : True,

    #Dataset configuraion
    'VOCAB_SIZE': 28, # Include 26 English Alphabet and <s> and </s>
    'SEQ_LENGTH': 40, # Output length would be sequence length plus key length
    'OUTPUT_LAYERS':-1, # Todo

    #Enigma configuraion
    'ROTOR': 'II IV V',
    'REFLECTOR': 'B',
    'RING_SETTING': [1, 20, 11],
    'PLUGBOARD': 'AV BS CG DL FU HZ IN KM OW RX',

    # Loss configuration
    'LOSS_TYPE': 'POS',

    # Log and checkpoint
    'LOG': os.path.join('tensorboard', strftime("%a%d%b%Y%H%M%S", gmtime())),
    'LOAD_CKPT': None, # Paste the path of checkpoint here

    # Batch mode
    'PROGRESS_BAR': 1, # Turn this off if the programme shows reprinting issues in batch mode output.
}