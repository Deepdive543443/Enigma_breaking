from config import args
from train import train
from utils import launch_tensorboard, save_checkpoint, load_checkpoint

import torch
optim = torch.optim
from model import Encoder, cp_2_key_model
from dataset import Enigma_simulate_c_2_p, Enigma_simulate_cp_2_k_limited, Enigma_simulate_cp_2_k
from torch.utils.data import DataLoader
import argparse


if __name__ == '__main__':
    # config arguments
    ap = argparse.ArgumentParser()
    for k, v in args.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)

    args = vars(ap.parse_args())

    # Print the configuration
    print("Config: ")
    for k, v in args.items():
        print(f"{k}: {v}")

    # Launch tensorboard
    if args['TENSORBOARD'] == 1:
        url = launch_tensorboard('tensorboard')
        print(f"Tensorboard listening on {url}")



    # Setting dataset and loader
    if args['TYPE'] == 'Encoder':
        dataset = Enigma_simulate_c_2_p(args=args)
    elif args['TYPE'] == 'CP2K' or 'CP2K_RNN':
        dataset = Enigma_simulate_cp_2_k_limited(args=args)
        # dataset = Enigma_simulate_cp_2_k(args=args)



    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args['BATCH_SIZE'],
        collate_fn=dataset.collate_fn_padding,
        shuffle=True,
        num_workers=args['NUM_WORKERS']
    )


    # Configure model
    if args['LOAD_CKPT'] is None:
        # Start training from scratch
        if args['TYPE'] == 'Encoder':
            # Training a new Encoder
            model = Encoder(args=args)
        elif args['TYPE'] == 'CP2K' or 'CP2K_RNN':
            if args['PRE_TRAINED_ENC'] is not None:
                # Start training on a pretrained Encoder
                pretrained_enc, _, _ = load_checkpoint(args['PRE_TRAINED_ENC'])
                model = cp_2_key_model(args=args, out_channels=dataset.tags_num(), pre_trained_encoder=pretrained_enc)
            else:
                #Initialize a new model
                model = cp_2_key_model(args=args, out_channels=dataset.tags_num(), pre_trained_encoder=None)

        model.to(args['DEVICE'])
        optimizer = optim.Adam(params=model.parameters(), lr=args['LEARNING_RATE'], betas=(args['BETA1'], args['BETA1']), eps=args['EPS'])
    else:
        # Continue training on previous weights and optimizer setting
        # This would also overwrite the current args by the one
        model, optimizer, _ = load_checkpoint(args['LOAD_CKPT'])

    # Use Torch.compile()
    # This features require pytorch 2.0
    if args['USE_COMPILE'] != 0:
        model = torch.compile(model)



    # Training loop
    print('\nStart training...\n')
    train(
        model=model,
        optimizer=optimizer,
        dataset=dataset,
        dataloader=dataloader,
        args=args
    )



