from config import args
from train import train
from utils import launch_tensorboard, save_checkpoint, load_checkpoint

import torch
optim = torch.optim
from model import Encoder, cp_2_key_model
from dataset import Enigma_simulate_c_2_p, Enigma_simulate_cp_2_k
from torch.utils.data import DataLoader
import argparse


if __name__ == '__main__':
    # config arguments
    ap = argparse.ArgumentParser()
    for k, v in args.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)

    args = vars(ap.parse_args())

    # Launch tensorboard
    url = launch_tensorboard('tensorboard')
    print(f"Tensorboard listening on {url}")

    # Setting dataset and loader
    if args['TYPE'] == 'Encoder':
        dataset = Enigma_simulate_c_2_p(args=args)
    elif args['TYPE'] == 'CP2K':
        dataset = Enigma_simulate_cp_2_k(args=args)



    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args['BATCH_SIZE'],
        collate_fn=dataset.collate_fn_padding,
        shuffle=True
    )


    # Configure model
    if args['LOAD_CKPT'] is None:
        # Start training from scratch
        if args['TYPE'] == 'Encoder':
            # Training a new Encoder
            model = Encoder(args=args)
        elif args['TYPE'] == 'CP2K':
            if args['PRE_TRAINED_ENC'] is not None:
                # Start training on a pretrained Encoder
                pretrained_enc, _, _ = load_checkpoint(args['PRE_TRAINED_ENC'])
                model = cp_2_key_model(args=args, out_channels=len(dataset), pre_trained_encoder=pretrained_enc)
            else:
                #Initialize a new model
                model = cp_2_key_model(args=args, out_channels=len(dataset), pre_trained_encoder=None)

        model.to(args['DEVICE'])
        optimizer = optim.Adam(params=model.parameters(), lr=args['LEARNING_RATE'], betas=(args['BETA1'], args['BETA1']), eps=args['EPS'])
    else:
        # Continue training on previous weights and optimizer setting
        # This would also overwrite the current args by the one
        model, optimizer, _ = load_checkpoint(args['LOAD_CKPT'])

    # Print lr strategy
    print(f"Warm up to: {args['LEARNING_RATE'] * min(pow(args['WARMUP_STEP'], -0.5), args['WARMUP_STEP'] * pow(args['WARMUP_STEP'], -1.5))}")


    # Training loop



    train(
        model=model,
        optimizer=optimizer,
        dataset=dataset,
        dataloader=dataloader,
        args=args
    )


