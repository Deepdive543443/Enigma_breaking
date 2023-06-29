from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from enigma.machine import EnigmaMachine
from enigma.rotors.rotor import Rotor
from enigma.plugboard import Plugboard
from itertools import product
import numpy as np
import torch
from config import args


class Enigma_simulate_dataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.enigma_machine = EnigmaMachine.from_key_sheet(
            rotors='II IV V',
            reflector='B',
            ring_settings=[1, 20, 11],
            plugboard_settings='AV BS CG DL FU HZ IN KM OW RX'
        )

        self.seq_length = args['SEQ_LENGTH']

        indice = [i for i in range(26)]
        self.initial_state = list(product(indice, indice, indice))

        self.char_to_indice = {chr(ord('A') + i): i for i in range(26)} # This gives a dict with {alphabet: index}
        self.char_to_indice['<s>'], self.char_to_indice['</s>'] = 26, 27
        self.indice_to_char = {v: k for k, v in self.char_to_indice.items()}

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

        return torch.cat([start_token, cipher_text_indice,end_token, plaintext_indice]), \
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
    def __init__(self, args, mode='train'):
        super().__init__(args)
        # Using limited key with number of sample being set
        # self.key_step = keys_step
        self.initial_state = self.initial_state[args['LIMITED_KEYS_START']:args['LIMITED_KEYS_END']:args['LIMITED_KEYS_STEP']]  #* sample_num
        # self.initial_state_2_idx = {state:idx for idx, state in enumerate(self.initial_state)}

        # Given different number for training and testing
        self.initial_state_test = self.initial_state * (args['SAMPLES_PER_KEYS'] // 2)
        self.initial_state_train = self.initial_state * args['SAMPLES_PER_KEYS']

        # Initial state dictionary
        self.initial_state_dict = {i: state for i, state in enumerate(self.initial_state)}

        # Setting up mode
        self.initial_state = self.initial_state_train if mode == 'train' else self.initial_state_test

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

        return torch.cat([start_token, cipher_text_indice,end_token, plaintext_indice]), initial_position_indice
               #torch.LongTensor([self.initial_state_2_idx[self.initial_state[index]]])

    def tags_num(self):
        return 26

    def collate_fn_padding(self, batch):
        # Each item in batch (input, target)
        inputs_batch = []
        targets_batch = []

        for cipher, plain in batch:
            inputs_batch.append(cipher)
            targets_batch.append(plain)

        # Merge list of sentences into a batch with padding
        batch_inputs = pad_sequence(inputs_batch, padding_value=0).T
        batch_targets = torch.stack(targets_batch)

        # Transfer back to [batch, indice]
        return batch_inputs, batch_targets

    def getsample(self):
        index = int(torch.randint(low=0, high=len(self.initial_state), size=[]))
        c, p = self.__getitem__(index)
        c, p = c.unsqueeze(0), p.unsqueeze(0)
        return c, p
        
    




if __name__ == "__main__":
    from config import args
    from model import cp_2_key_model
    from torch.utils.data import DataLoader

    args['TYPE'] = 'CP2K_RNN_ENC'

    dataset = Enigma_simulate_cp_2_k_limited(args=args)
    # length = len(dataset.initial_state)
    # print(length)
    # print(dataset.initial_state)
    # print(len(dataset.initial_state))
    # print(dataset.initial_state[::length // 3])
    # print(len(dataset.initial_state[::10]))
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=6,
        collate_fn=dataset.collate_fn_padding,
        shuffle=True
    )
    print(dataset.tags_num())
    model = cp_2_key_model(args=args, out_channels=26)
    for inputs, targets in dataloader:
        print(inputs.shape, targets.shape, targets)
        # preds = model(inputs)
        # print(preds.shape)
        # print((torch.argmax(preds, dim=2).T).shape)
        break

    for inputs, targets in dataloader:
        print(inputs.shape, targets.shape, targets)
        break
    # dataset2 = Enigma_simulate_dataset(args=args)
    # print(len(dataset2.initial_state))
    # print(len(dataset2.initial_state) // 4)
    # print(dataset2.initial_state[::len(dataset2.initial_state) // 4])

   