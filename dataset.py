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


        # return torch.cat([
        #     start_token,
        #     cipher_text_indice,
        #     end_token,
        #     start_token,
        #     plaintext_indice,
        #     end_token,
        # ]), torch.cat([start_token, initial_position_indice, end_token])

        return torch.cat([
            start_token,
            cipher_text_indice,
            end_token,
            start_token,
            initial_position_indice,
            end_token,
        ]), torch.cat([start_token, plaintext_indice, end_token])

    def getsample(self):
        index = int(torch.randint(low=0, high=len(self.initial_state), size=[]))
        sample_input, sample_target = self.__getitem__(index)
        sample_input, sample_target = sample_input.unsqueeze(0), sample_target.unsqueeze(0)
        return sample_input, sample_target


def collate_fn_padding(batch):
    # Each item in batch (input, target)
    inputs_batch = []
    target_batch = []

    for input, target in batch:
        inputs_batch.append(input)
        target_batch.append(target)
        # teaching_batch.append(target[:-1])

    # Merge list of sentences into a batch with padding
    batch_inputs = pad_sequence(inputs_batch, padding_value=0).T
    batch_target = pad_sequence(target_batch, padding_value=0).T

    # Transfer back to [batch, indice]
    return batch_inputs, batch_target
    
        
    




if __name__ == "__main__":
    from model import Transformer
    enigma_dataset = Enigma_simulate_dataset(args=args)
    cipher_text_indice, plaintext_indice = enigma_dataset.__getitem__(0)
    print(cipher_text_indice, plaintext_indice)

    enigma_dataloader = DataLoader(enigma_dataset, batch_size=3, shuffle=True, drop_last=True)

    for idx, (inputs, targets) in enumerate(enigma_dataloader):
        print(f"Input shape: {inputs.shape}  Target shape: {targets.shape}")
        model = Transformer(args=args)
        outputs = model(inputs, targets)
        print(f"Output shape: {outputs.shape}")
        break

   