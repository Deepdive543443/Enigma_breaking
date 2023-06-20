import os
from time import gmtime, strftime

args = {
    # Training hyperparameters
    'DEVICE': 'cuda',
    'BATCH_SIZE' : 256,
    'LEARNING_RATE' : 3e-4,  #512 ** (-0.5),# ,
    'BETA1': 0.9,
    'BETA2': 0.98,
    'EPS': 1e-9,
    'EPOCHS' : 1000,
    'WARMUP_STEP': 2000,
    'NUM_WORKERS': 1,

    # Model configuraion
    'TYPE': 'CP2K_RNN', # 'Encoder' for sequence labeling Or 'CP2K' for text pair classification 'CP2K_RNN' 'CP2K_RNN_ENC'
                        # 'CP2K_RNN' for text pair classification with RNN. 'CP2K_RNN_ENC' for text pair classification with RNN and transformer encoder

    # RNN's configuration
    'LAYERS' : 2,
    'EMB_DIM' : 300,
    'HIDDEN': 300,
    'DROPOUT': 0.5,
    'RNN_TYPE' : 'LSTM',
    'BIDIRECTION' : True,

    # Transformer encoder configuration
    'ENC_LAYERS' : 2,
    'FEED_DIM': 1200,
    'ATTN_HEAD': 6,
    # Pretrained setting


    'PRE_TRAINED_ENC': None,


    #Dataset configuraion
    'VOCAB_SIZE': 28, # Include 26 English Alphabet and <s> and </s>
    'SEQ_LENGTH': 40, # Output length would be sequence length plus key length
    'LIMITED_KEYS_START': 0,
    'LIMITED_KEYS_END': -1, # 17576 if using all key pairs
    'LIMITED_KEYS_STEP': int(17576 // 40), # int(17576 // num_key_pairs)
    'SAMPLES_PER_KEYS': 800,

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

    # Batch mode(cluster mode)
    'USE_COMPILE': 0,
    'PROGRESS_BAR': 1, # Turn this off if the programme shows reprinting issues in batch mode output.
    'TEST': 0,
    'TENSORBOARD': 0
}