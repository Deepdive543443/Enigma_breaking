from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from enigma.machine import EnigmaMachine
from itertools import product
import random
import torch


class Enigma_simulate_dataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.enigma_machine = EnigmaMachine.from_key_sheet(
            rotors='II IV V',
            reflector='B',
            ring_settings=[1, 20, 11],
            plugboard_settings='AV BS CG DL FU HZ IN KM OW RX',

        )

        self.seq_length = args['SEQ_LENGTH']

        indice = [i for i in range(26)]
        self.initial_state = list(product(indice, indice, indice))

        self.char_to_indice = {chr(ord('A') + i): i for i in range(26)} # This gives a dict with {alphabet: index}
        self.char_to_indice['<s>'], self.char_to_indice['</s>'] = 26, 27
        self.indice_to_char = {v: k for k, v in self.char_to_indice.items()}

    def get_cipher_plain_positions(self, index, length):
        # Generate the string of plaintext
        plaintext_indice = torch.randint(low=0, high=25, size=[length])
        plaintext = ''.join([self.indice_to_char[int(char)] for char in plaintext_indice])

        # Obtain the initial state by index
        initial_position_indice = self.initial_state[index]
        initial_position = ''.join([self.indice_to_char[int(char)] for char in initial_position_indice])

        # Setting initial position for Enigma machine
        self.enigma_machine.set_display(initial_position)

        # obtain ciphertext and all matched states
        ciphertext = ''
        states = [initial_position]
        for char in plaintext:
            ciphertext += self.enigma_machine.process_text(char)
            states.append(self.enigma_machine.get_display())

        # Transfer ciphertext and states to indice
        cipher_text_indice = torch.LongTensor([self.char_to_indice[char] for char in ciphertext])
        states_indice = torch.LongTensor([[self.char_to_indice[char] for char in state] for state in states[:-1]])
        return plaintext_indice, cipher_text_indice, states_indice, plaintext, ciphertext, states


    def __len__(self):
        return len(self.initial_state)

    def tags_num(self):
        return self.__len__()

    def getsample(self):
        index = int(torch.randint(low=0, high=len(self.initial_state), size=[]))
        c, p = self.__getitem__(index)
        c, p = c.unsqueeze(0), p.unsqueeze(0)
        return c, p

    def collate_fn_padding(self, batch):
        # Each item in batch (input, target)
        cipher_batch = []
        plain_batch = []

        for cipher, plain in batch:
            cipher_batch.append(cipher)
            plain_batch.append(plain)

        # Merge list of sentences into a batch with padding
        batch_ciphers = pad_sequence(cipher_batch, padding_value=0).T
        batch_plains = pad_sequence(plain_batch, padding_value=0).T

        # Transfer back to [batch, indice]
        return batch_ciphers, batch_plains

class Enigma_simulate_c_2_p(Enigma_simulate_dataset):
    def __init__(self, args):
        super(Enigma_simulate_c_2_p, self).__init__(args)

    def __getitem__(self, index):
        # Generate a sequence randomly
        plaintext_indice = torch.randint(low=0, high=25, size=[self.seq_length])
        plaintext = ''.join([self.indice_to_char[int(char)] for char in plaintext_indice])
        # print(plaintext)

        # Obtain the intial position from product 
        initial_position_indice  = torch.LongTensor(self.initial_state[index])
        initial_position = ''.join([self.indice_to_char[int(char)] for char in initial_position_indice])
        # initial_position_indice = [ for char in initial_position]

        # setting the enigma 
        self.enigma_machine.set_display(initial_position)

        # translation the
        cipher_text = self.enigma_machine.process_text(plaintext)
        cipher_text_indice = torch.LongTensor([self.char_to_indice[char] for char in cipher_text])

        # Outputs in forms of [keys, ]
        start_token = torch.LongTensor([26])
        end_token = torch.LongTensor([27])

        return torch.cat([
            start_token,
            cipher_text_indice,
            end_token,
            initial_position_indice,
        ]), torch.cat([
            start_token,
            plaintext_indice,
            end_token,
            initial_position_indice,
        ])

class Enigma_simulate_cp_2_k(Enigma_simulate_dataset):
    def __init__(self, args):
        super(Enigma_simulate_cp_2_k, self).__init__(args)
        # self.initial_state = list(product(indice, indice, indice))
        self.initial_state_dict = {i: state for i, state in enumerate(self.initial_state)}

    def __getitem__(self, index):
        # Generate a sequence randomly
        plaintext_indice = torch.randint(low=0, high=25, size=[self.seq_length])
        plaintext = ''.join([self.indice_to_char[int(char)] for char in plaintext_indice])
        # print(plaintext)

        # Obtain the intial position from product
        initial_position_indice = torch.LongTensor(self.initial_state[index])
        initial_position = ''.join([self.indice_to_char[int(char)] for char in initial_position_indice])
        # initial_position_indice = [ for char in initial_position]

        # setting the enigma
        self.enigma_machine.set_display(initial_position)

        # translation the
        cipher_text = self.enigma_machine.process_text(plaintext)
        cipher_text_indice = torch.LongTensor([self.char_to_indice[char] for char in cipher_text])

        # Outputs in forms of [keys, ]
        start_token = torch.LongTensor([26])
        end_token = torch.LongTensor([27])

        return torch.cat([cipher_text_indice, plaintext_indice]), \
               torch.LongTensor([index])


    def collate_fn_padding(self, batch):
        # Each item in batch (input, target)
        inputs_batch = []
        targets_batch = []

        for cipher, plain in batch:
            inputs_batch.append(cipher)
            targets_batch.append(plain)

        # Merge list of sentences into a batch with padding
        batch_inputs = pad_sequence(inputs_batch, padding_value=0).T
        batch_targets = torch.cat(targets_batch)

        # Transfer back to [batch, indice]
        return batch_inputs, batch_targets

    def getsample(self):
        index = int(torch.randint(low=0, high=len(self.initial_state), size=[]))
        c, p = self.__getitem__(index)
        c = c.unsqueeze(0)
        return c, p


class Enigma_simulate_cp_2_k_limited(Enigma_simulate_dataset):
    def __init__(self, args, mode='train', seed=114514):
        super().__init__(args)
        # Using limited key with number of sample being set
        self.initial_state = self.initial_state[args['LIMITED_KEYS_START']:args['LIMITED_KEYS_END']:args['LIMITED_KEYS_STEP']]  #* sample_num

        # Given different number for training and testing
        self.initial_state_test = self.initial_state * 2
        self.initial_state_train = self.initial_state * args['SAMPLES_PER_KEYS']

        # Initial state dictionary
        self.initial_state_dict = {i: state for i, state in enumerate(self.initial_state)}
        self.initial_state_dict_reverse = {v: k for k, v in self.initial_state_dict.items()}

        # Setting up mode
        self.initial_state = self.initial_state_train if mode == 'train' else self.initial_state_test

        # Fix seed if dataset is in test mode
        if mode == 'test':
            torch.manual_seed(seed)
            random.seed(seed)

    def setMode(self, mode='test'):
        '''
        Given less amount of sample when testing


        :param mode:
        :return:
        '''
        if mode == 'test':
            self.initial_state = self.initial_state_test
        elif mode == 'train':
            self.initial_state = self.initial_state_train

    def __getitem__(self, index):
        # Generate a random sequence with random length
        rand_length = random.randint(self.seq_length[0], self.seq_length[1])

        plaintext_indice, cipher_text_indice, states_indice, plaintext, ciphertext, states = self.get_cipher_plain_positions(index, rand_length)

        # Transfer indices to vectors
        plaintext_text_vector = torch.zeros(plaintext_indice.shape[0], 26)
        plaintext_text_vector[torch.arange(plaintext_indice.shape[0]), plaintext_indice] = 1

        cipher_text_vector = torch.zeros(cipher_text_indice.shape[0], 26)
        cipher_text_vector[torch.arange(cipher_text_indice.shape[0]), cipher_text_indice] = 1

        mask = torch.zeros(rand_length)


        return torch.cat([cipher_text_vector, plaintext_text_vector], dim=1), states_indice, mask
               #torch.LongTensor([self.initial_state_2_idx[self.initial_state[index]]])

    def tags_num(self):
        return 26

    def collate_fn_padding(self, batch):
        # Each item in batch (input, target)
        inputs_batch = []
        targets_batch = []
        mask_batch = []

        for inputs, targets, masks in batch:
            inputs_batch.append(inputs)
            targets_batch.append(targets)
            mask_batch.append(masks)

        # Merge list of sentences into a batch with padding

        inputs_batch = pad_sequence(inputs_batch)
        targets_batch = pad_sequence(targets_batch, batch_first=True).permute(2, 1, 0) # [seq, rotor] -> [batch, seq, rotor] -> [rotor, seq, batch]
        mask_batch = pad_sequence(mask_batch, batch_first=True, padding_value=1).bool() # -> [seq, batch]

        # Transfer back to [batch, indice]
        return inputs_batch, targets_batch, mask_batch

    def getsample(self):
        index = int(torch.randint(low=0, high=len(self.initial_state), size=[]))
        c, p = self.__getitem__(index)
        c, p = c.unsqueeze(0), p.unsqueeze(0)
        return c, p
        

def cipher_plain_text_2_tensor(ciphertext, plaintext):
    # Length checking

    if len(ciphertext) != len(plaintext):
        raise Exception("ciphertext's length needs to match with plaintext")

    # Dictionary for transfering char to index
    char_to_indice = {chr(ord('A') + i): i for i in range(26)}


    # transfer both text into list of integers
    ciphertext_indice = torch.LongTensor([char_to_indice[char] for char in ciphertext.upper()])
    plaintext_indice = torch.LongTensor([char_to_indice[char] for char in plaintext.upper()])

    # Transfer indices into one-hot vectors
    plaintext_text_vector = torch.zeros(plaintext_indice.shape[0], 26)
    plaintext_text_vector[torch.arange(plaintext_indice.shape[0]), plaintext_indice] = 1

    cipher_text_vector = torch.zeros(ciphertext_indice.shape[0], 26)
    cipher_text_vector[torch.arange(ciphertext_indice.shape[0]), ciphertext_indice] = 1

    # Mask that useless during testing.
    mask = torch.zeros(len(ciphertext))
    return torch.cat([cipher_text_vector, plaintext_text_vector], dim=1).unsqueeze(1), mask.unsqueeze(0)


class Random_setting_dataset(Enigma_simulate_dataset):
    def __init__(self, args):
        super(Random_setting_dataset, self).__init__(args)
        self.batchsize = args['BATCH_SIZE']
        self.args = args
        self.rotors_roman = {
            0:'I', 1:'II', 2:'III', 3:'IV', 4: 'V', 5:'VI', 6:'VII', 7:'VIII'
        }

        self.num_key_sheet = 0
        if args['Random_rotors_num'] is not None:
            self.num_key_sheet += 1
        # random reflector (B and C) shape : [1]
        if args['Random_reflector'] is not None:
            self.num_key_sheet += 1
        # random ring setting (26 ringsetting for each rotor) shape : [1 - 3, 26]
        if args['Random_ringsetting_num'] is not None:
            self.num_key_sheet += 1
        # Random plugboard( 26x 26 plugboard setting) shape : [26, 26]
        if args['Random_plugboard_num'] is not None:
            self.num_key_sheet += 1


    def make_key_sheet(self, args):
        key_sheet = {
            'rotors' : 'II IV V',
            'reflector' : 'B',
            'ring_settings' : [1, 20, 11],
            'plugboard_settings' : 'AV BS CG DL FU HZ IN KM OW RX'
        }

        targets = []

        # random rotors (8 different types of rotors) shape : [1 - 3, 8]
        if args['Random_rotors_num'] is not None:

            # Update keysheet
            rotors_str = key_sheet['rotors'].split(' ')
            randomized_rotor_int = [random.randint(0, 7) for i in range(args['Random_rotors_num'])]
            randomized_rotor = [self.rotors_roman[i] for i in randomized_rotor_int]
            rotors_str[:args['Random_rotors_num']] = randomized_rotor
            key_sheet['rotors'] = ' '.join(rotors_str)

            # To tensor
            targets.append(torch.LongTensor(randomized_rotor_int))



        # random reflector (B and C) shape : [1]
        if args['Random_reflector'] is not None:

            # Update keysheet
            reflector_indice = torch.randint(0, 2, ())
            key_sheet['reflector'] = 'B' if reflector_indice == 0 else 'C'

            # Adding target
            targets.append(reflector_indice)


        # random ring setting (26 ringsetting for each rotor) shape : [1 - 3, 26]
        if args['Random_ringsetting_num'] is not None:

            # Update keysheet
            new_settings = [random.randint(0, 25) for i in range(args['Random_ringsetting_num'])]
            # key_sheet['ring_settings'][:-args['Random_ringsetting_num']] = new_settings
            for idx, setting in enumerate(new_settings):
                key_sheet['ring_settings'][idx] = setting

            # Adding target
            targets.append(torch.LongTensor(new_settings))



        # Random plugboard( 26x 26 plugboard setting) shape : [26, 26]
        if args['Random_plugboard_num'] is not None:
            # Generate the new connection
            plugs = torch.randperm(26)[:args['Random_plugboard_num'] * 2]

            # transfer to string
            plug_board_str = []

            for idx, connection in zip(plugs[::2], plugs[1::2]):
                plug_board_str.append(chr(idx + 65) + chr(connection + 65))
            key_sheet['plugboard_settings'] = ' '.join(plug_board_str)

            # to tensor
            original_connection = torch.linspace(0, 25, steps=26).long()
            original_connection[plugs[::2]] = plugs[1::2]
            original_connection[plugs[1::2]] = plugs[::2]

            targets.append(original_connection)


        return key_sheet, targets

    def __len__(self):
        return self.batchsize * 500

    def collate_fn_padding(self, batch):
        # Each item in batch (input, target)
        inputs_batch = []
        targets_batch = []
        mask_batch = []

        key_sheet_batch = [[] for i in range(self.num_key_sheet)]

        for inputs, targets, masks, key_sheet in batch:
            inputs_batch.append(inputs)
            targets_batch.append(targets)
            mask_batch.append(masks)

            for idx, key in enumerate(key_sheet):
                key_sheet_batch[idx].append(key)

        # Merge list of sentences into a batch with padding

        inputs_batch = pad_sequence(inputs_batch)
        targets_batch = pad_sequence(targets_batch, batch_first=True).permute(2, 1, 0) # [seq, rotor] -> [batch, seq, rotor] -> [rotor, seq, batch]
        mask_batch = pad_sequence(mask_batch, batch_first=True, padding_value=1).bool() # -> [seq, batch]

        for idx, key in enumerate(key_sheet_batch):
            key_sheet_batch[idx] = torch.stack(key)


        # Transfer back to [batch, indice]
        return inputs_batch, targets_batch, mask_batch, key_sheet_batch

    def __getitem__(self, index):
        # Generate a random sequence with random length
        rand_length = random.randint(self.seq_length[0], self.seq_length[1])
        rand_states = random.randint(0, 17575)

        key_sheet_random, targets = self.make_key_sheet(self.args)
        self.enigma_machine = EnigmaMachine.from_key_sheet(
            rotors=key_sheet_random['rotors'],
            reflector=key_sheet_random['reflector'],
            ring_settings=key_sheet_random['ring_settings'],
            plugboard_settings=key_sheet_random['plugboard_settings'],

        )

        plaintext_indice, cipher_text_indice, states_indice, plaintext, ciphertext, states = self.get_cipher_plain_positions(rand_states, rand_length)

        # Transfer indices to vectors
        plaintext_text_vector = torch.zeros(plaintext_indice.shape[0], 26)
        plaintext_text_vector[torch.arange(plaintext_indice.shape[0]), plaintext_indice] = 1

        cipher_text_vector = torch.zeros(cipher_text_indice.shape[0], 26)
        cipher_text_vector[torch.arange(cipher_text_indice.shape[0]), cipher_text_indice] = 1

        mask = torch.zeros(rand_length)
        sas = torch.cat([cipher_text_vector, plaintext_text_vector], dim=1)

        return torch.cat([cipher_text_vector, plaintext_text_vector], dim=1), states_indice, mask, targets

