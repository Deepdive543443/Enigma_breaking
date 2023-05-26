from config import args
from train import train
from utils import launch_tensorboard

import torch
optim = torch.optim
from model import RNN
from dataset import Enigma_simulate_dataset, collate_fn_padding
from torch.utils.data import DataLoader

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # Launch tensorboard
    url = launch_tensorboard('tensorboard')
    print(f"Tensorboard listening on {url}")


    # Setting dataset and loader
    dataset = Enigma_simulate_dataset(args=args)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args['BATCH_SIZE'],
        collate_fn=collate_fn_padding,
        shuffle=True
    )


    # Setting RNN model
    model = RNN(args=args)
    model.to(args['DEVICE'])

    # Training loop
    train(
        model=model,
        dataloader=dataloader,
        args=args
    )


