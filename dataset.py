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

        # Obtain the cipher text from enigma
        cipher_text = self.enigma_machine.process_text(plaintext)
        # print(cipher_text)
        cipher_text_indice = torch.LongTensor([self.char_to_indice[char] for char in cipher_text])
        return torch.cat([initial_position_indice, cipher_text_indice]), torch.cat([initial_position_indice, plaintext_indice])

def collate_fn_padding(batch):
    # Each item in batch (input, target)
    inputs_batch = []
    target_batch = []

    for input, target in batch:
        inputs_batch.append(input)
        target_batch.append(target)
        # teaching_batch.append(target[:-1])

    # Merge list of sentences into a batch with padding
    batch_inputs = pad_sequence(inputs_batch, padding_value=0).permute(1, 0, 2).squeeze(-1)
    batch_target = pad_sequence(target_batch, padding_value=0).permute(1, 0, 2).squeeze(-1) # [seq, batch, unsqueezed]

    # Transfer back to [batch, indice]
    return batch_inputs, batch_target
    
        
    




if __name__ == "__main__":
    from model import RNN
    enigma_dataset = Enigma_simulate_dataset(args=args)
    cipher_text_indice, plaintext_indice = enigma_dataset.__getitem__(0)
    print(cipher_text_indice, plaintext_indice)

    enigma_dataloader = DataLoader(enigma_dataset, batch_size=3, shuffle=True, drop_last=True)

    for idx, (inputs, targets) in enumerate(enigma_dataloader):
        print(f"Input shape :{inputs.shape}  Targets shape: {targets.shape}")
        model = RNN(args=args)
        outputs = model(inputs)
        print(f"Output shape: {outputs.shape}")
        break
    # enigma = EnigmaMachine.from_key_sheet(
    #     rotors='II IV V',
    #     reflector='B',
    #     ring_settings=[1, 20, 11],
    #     plugboard_settings='AV BS CG DL FU HZ IN KM OW RX'
    # )

    # enigma.set_display('AAA')

    # plaintext = 'IMADOG'
    # ciphertext = enigma.process_text(plaintext)
    # print(f'plaintext: {plaintext}\nProcessed text: {ciphertext}')

    # print(enigma.get_display())
    # enigma.set_display('AAA')
    # print(f'cipher text: {ciphertext}\nDeciphered text: {enigma.process_text(ciphertext)}')

    # rotor_1 = Rotor(
    #     model_name='Rotor 1', wiring='BDFHJLCPRTXVZNYEIWGAKMUSQO', ring_setting=0, stepping="Z"
    # )
    # input = 'A'
    # postion = ord(input) - ord('A') 

    # print(f'order:{postion}  {rotor_1.signal_in(postion)}')
    # rotor_1.rotate()
    # print(f'order:{postion}  {rotor_1.signal_in(postion)}')

    # indice = {i: chr(ord('A') + i) for i in range(26)}
    # plaintext_indice = torch.randint(low=0, high=25, size=[26])
    # for i in plaintext_indice:
    #     print(indice[int(i)])

    # print(torch.LongTensor((20, 23, 12)))


