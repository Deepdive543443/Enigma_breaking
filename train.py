import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from utils import TensorboardLogger


def train(model, dataloader, args):
    '''
    config the training pipeline.


    :param model:
    :param dataloader:
    :param args:
    :return:
    '''
    # Set up tensorboard logger
    logger = TensorboardLogger()

    # Setting progress bar
    # bar = tqdm(enumerate(dataloader), leave=True)

    # Setting the optimizer
    optimizer = optim.Adam(params=model.parameters(), lr=args['LEARNING_RATE'])
    mix_scaler = torch.cuda.amp.GradScaler()

    # Set up the loss
    critic = nn.CrossEntropyLoss()

    # Keep the best model
    loss_best = 9999999
    acc_best = 0

    for epoch in range(args['EPOCHS']):
        loss_sum = 0
        bar = tqdm(dataloader, leave=True)
        for idx, (inputs, targets) in enumerate(bar):
            # send to device
            inputs = inputs.to(args['DEVICE'])
            targets = targets.to(args['DEVICE'])

            # Make prediction
            with torch.cuda.amp.autocast():
                outputs = (model(inputs)).permute(1, 2, 0) # [seq, batch, features] -> [batch, features, seq]
                loss = critic(outputs, targets)

            # Update weights in mix precision
            optimizer.zero_grad()
            mix_scaler.scale(loss).backward()
            mix_scaler.step(optimizer)
            mix_scaler.update()

            # Bar
            bar.set_description(f"Epoch[{epoch + 1}/{args['EPOCHS']}]")

            # Logging per iter
            loss_sum += loss.item()
            logger.update('loss_batch', loss.item())
            logger.step()

        # Logg after each epoch
        acc, loss_test = test(model, dataloader, critic, args)
        logger.update('Accuracy', acc)
        logger.update('Loss_test', loss_test)
        logger.step()

    return model, optimizer

def test(model, dataloader, critic, args):
    '''
    Calculate the accuracy and print some example

    :param model:
    :param dataloader:
    :param critic:
    :param args:
    :return:
    '''

    model.eval()
    true_positive = 0
    samples = 0
    loss_sum = 0
    for inputs,targets in dataloader:
        inputs = inputs.to(args['DEVICE'])
        targets = targets.to(args['DEVICE'])

        # Compute loss
        outputs = (model(inputs)).permute(1, 2, 0)  # [seq, batch, features] -> [batch, features, seq]
        loss = critic(outputs, targets)

        # Compute metrics
        outputs_indices = torch.argmax(outputs, dim=1)
        mask = outputs_indices == targets


        # Track performance
        true_positive += mask.sum()
        samples += mask.shape[0] * mask.shape[1]
        loss_sum += loss.item()

    acc = true_positive / samples
    loss_sum /= mask.shape[0]
    model.train()
    return acc, loss_sum




if __name__ == "__main__":
    outputs_test = torch.randn(3, 300, 26)
    print(torch.argmax(outputs_test, dim=1).shape)
    print(torch.argmax(outputs_test, dim=1))



