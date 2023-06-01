import os
from time import gmtime, strftime

args = {
    # Training
    'DEVICE': 'cuda',
    'BATCH_SIZE' : 128,
    'LEARNING_RATE' : 3e-4,
    'EPOCHS' : 1000,

    # Model configuraion
    'RNN_TYPE' : 'LSTM',
    'LAYERS' : 2,
    'EMB_DIM' : 256,
    'HIDDEN': 3000,
    'ATTN_HEAD': 8,
    'DROPOUT': 0.5,
    'BIDIRECTION' : True,

    #Dataset configuraion
    'VOCAB_SIZE': 28, # Include 26 English Alphabet and <s> and </s>
    'SEQ_LENGTH': 15, # Output length would be sequence length plus key length
    'KEY_INPUT': True,
    'OUTPUT_LAYERS':-1, # Todo

    #Enigma configuraion
    'ROTOR': 'II IV V',
    'REFLECTOR': 'B',
    'RING_SETTING': [1, 20, 11],
    'PLUGBOARD': 'AV BS CG DL FU HZ IN KM OW RX',

    # Loss configuration
    'LOSS_TYPE': 'CROSS_ENTROPY',

    # Log and checkpoint
    'LOG': os.path.join('tensorboard', strftime("%a%d%b%Y%H%M%S", gmtime())),
    'LOAD_CKPT': None, # Paste the path of checkpoint here

    # Batch mode
    'PROGRESS_BAR': 1, # Turn this off if the programme shows reprinting issues in batch mode output.
}