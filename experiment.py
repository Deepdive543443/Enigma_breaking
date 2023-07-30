import torch
import random
from dataset import Random_setting_dataset
from torch.utils.data import DataLoader
from config import args
from model import cp_2_k_mask

if __name__ == "__main__":
    args['Random_rotors_num'] = 2
    args['Random_reflector'] = True

    args['Random_ringsetting_num'] = 2
    args['Random_plugboard_num'] = 3
    args['DEVICE'] = 'cpu'
    dataset = Random_setting_dataset(args)

    print(dataset.make_key_sheet(args))
    loader = DataLoader(dataset, batch_size=12, shuffle=True, drop_last=True, collate_fn=dataset.collate_fn_padding)
    loader_iter = iter(loader)
    inputs, position, mask, fur = next(loader_iter)
    print(inputs.shape, position.shape, mask.shape)

    print(torch.zeros(size=[]))


