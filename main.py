from config import args
from train import train
from utils import launch_tensorboard, save_checkpoint, load_checkpoint

import torch
optim = torch.optim
from model import RNN
from dataset import Enigma_simulate_dataset, collate_fn_padding
from torch.utils.data import DataLoader


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
    if args['LOAD_CKPT'] is None:
        # Start training from scratch
        model = RNN(args=args)
        model.to(args['DEVICE'])
        optimizer = optim.Adam(params=model.parameters(), lr=args['LEARNING_RATE'])
    else:
        # Continue training on previous weights and optimizer setting
        # This would also overwrite the current args by the one
        model, optimizer, _ = load_checkpoint(args['LOAD_CKPT'])

    # Training loop
    train(
        model=model,
        optimizer=optimizer,
        dataset=dataset,
        dataloader=dataloader,
        args=args
    )

