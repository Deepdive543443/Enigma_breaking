import torch
from config import args
nn = torch.nn
from tqdm import tqdm

def train(model, critic, optimizer, dataloader, dataset, args, tensorboard_logger):
    for epoch in args['EPOCHS']:
        # Initial progress bar
        progress_bar = tqdm(enumerate(dataloader))

        # Initial loss logging 
        loss_log = 0


        for batch_idx, (inputs, targets) in progress_bar:
            # Sending all contents to GPU
            inputs = inputs.to(args['DEVICE'])
            targets = targets.to(args['DEVICE'])

            # Forwarding and compute loss
            outputs = model(inputs)
            loss = critic(outputs, targets)

            # Backward and optimizing 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Progress bar monitoring 
            progress_bar.set_description(f'Epochs: {epoch + 1}/ {args['EPOCHS']}')
            progress_bar.set_postfix(loss = loss.item(), loss_sum = loss_log)

            # Tensorboard logging


        # Saving checkpoint

def test():
    pass


def BLEU():
    pass

def print_example(input, target, prediction):
    pass