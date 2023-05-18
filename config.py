args = {
    # Training hyperparameters
    'DEVICE': 'cuda',
    'BATCH_SIZE' : 256,
    'LEARNING_RATE' : 3e-4,
    'EPOCHS' : 300,

    # Training setting
    'SAVE_CKPT': False,
    'LOAD_CKPT': False, # Type the path of checkpoint here



    # Model configuraion
    'RNN_TYPE' : 'GRU', # 'GRU'
    'LAYERS' : 2,
    'EMB_DIM' : 300,
    'HIDDEN': 1024,
    'DROPOUT': 0.5,
    'BIDIRECTION' : True,

    #Dataset
    'VOCAB_SIZE': 27, # Include 26 English Alphabet and a <BLANK> mark
    'SEQ_LENGTH': 30, # Output length would be sequence length plus key length
    'OUTPUT_LAYERS':-1,

    #Enigma
    'ROTOR': 'II IV V',
    'REFLECTOR': 'B',
    'RING_SETTING': [1, 20, 11],
    'PLUGBOARD': 'AV BS CG DL FU HZ IN KM OW RX'
}