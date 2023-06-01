import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from utils import TensorboardLogger, save_checkpoint
from math import pow


# def lr_scheduler(args, optimizer):
#     l = lambda current_step: min(pow(current_steps, -0.5), current_steps * pow(args['WARMUP_STEP'], 1.5))
#     sceduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=l)
#     return sceduler

def train(model, optimizer, dataset, dataloader, args):
    '''
    config the training pipeline.


    :param model:
    :param dataloader:
    :param args:
    :return:
    '''
    # Set up tensorboard logger
    logger = TensorboardLogger(args)

    # Setting the optimizer
    # optimizer = optim.Adam(params=model.parameters(), lr=args['LEARNING_RATE'])
    mix_scaler = torch.cuda.amp.GradScaler()

    # Set up the loss
    critic = nn.CrossEntropyLoss()

    # Tracking the training stats and learning rate scheduling
    loss_best = 9999999
    acc_best = 0
    current_steps = 1
    # l = lambda current_steps: min(1 / (current_steps ** 0.5), current_steps * (args['WARMUP_STEP'] ** 1.5))
    # sceduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=[l])

    for epoch in range(args['EPOCHS']):
        if args['PROGRESS_BAR'] == 1:
            bar = tqdm(dataloader, leave=True)
        else:
            bar = dataloader

        for idx, (inputs, targets) in enumerate(bar):
            # send to device
            inputs = inputs.to(args['DEVICE'])
            targets = targets.to(args['DEVICE'])

            # Make prediction
            with torch.cuda.amp.autocast():
                # outputs = (model.recursive(inputs, args['DEVICE'])).permute(1, 2, 0)
                outputs = (model(inputs, targets)).permute(1, 2, 0) # [seq, batch, features] -> [batch, features, seq]
                loss = critic(outputs, targets[:, 1:])

            # Update weights in mix precision
            optimizer.zero_grad()
            mix_scaler.scale(loss).backward()
            mix_scaler.step(optimizer)
            mix_scaler.update()

            # Learning rate scheduling
            # for g in optimizer.param_groups:
            #     g['lr'] = args['LEARNING_RATE'] * min(pow(current_steps, -0.5), current_steps * pow(args['WARMUP_STEP'], -1.5))
            current_steps += 1

            # Bar
            bar.set_description(f"Epoch[{epoch + 1}/{args['EPOCHS']}]") if args['PROGRESS_BAR'] == 1 else None
            bar.set_postfix(lr=optimizer.state_dict()['param_groups'][0]['lr'], steps=current_steps) if args['PROGRESS_BAR'] == 1 else None

            # Logging per iter
            logger.update('loss_batch', loss.item())
            logger.update('lr', optimizer.state_dict()['param_groups'][0]['lr'])

        # Logg after each epoch
        with torch.no_grad():
            acc, loss_test = test(model, dataset, dataloader, critic, epoch, args)


        # Saving the checkpoint
        if loss_best >= loss_test:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                args=args,
                filename='lowest_loss.pt',
                info=f"Epoch: {epoch + 1} Acc: {acc} Loss_avg: {loss_test}"
            )
            loss_best = loss_test

        if acc_best <= acc:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                args=args,
                filename='best_acc.pt',
                info=f"Epoch: {epoch + 1} Acc: {acc} Loss_avg: {loss_test}"
            )
            acc_best = acc


        # Logging per epoch
        logger.update('Accuracy', acc)
        logger.update('Loss_test', loss_test)
        # logger.step()

    return model, optimizer

def test(model, dataset, dataloader, critic, epoch, args):
    '''
    Calculate the accuracy and print some example

    :param model:
    :param dataloader:
    :param critic:
    :param args:
    :return:
    '''

    model.eval()

    # Tracking process
    true_positive = 0
    samples = 0
    loss_sum = 0

    # Training loop
    for inputs, targets in dataloader:
        inputs = inputs.to(args['DEVICE'])
        targets = targets.to(args['DEVICE'])

        # Compute loss
        outputs = (model(inputs, targets)).permute(1, 2, 0)  # [seq, batch, features] -> [batch, features, seq]
        # outputs = (model.recursive(inputs, args['DEVICE'], target_length=targets[:, 1:].shape[1])).permute(1, 2, 0)
        loss = critic(outputs, targets[:, 1:])

        # Compute metrics
        outputs_indices = torch.argmax(outputs, dim=1)
        mask = outputs_indices == targets[:, 1:]


        # Track performance
        true_positive += mask.sum()
        samples += mask.shape[0] * mask.shape[1]
        loss_sum += loss.item()

    # print example
    sample_input, sample_target = dataset.getsample()
    sample_input, sample_target = sample_input.to(args['DEVICE']), sample_target.to(args['DEVICE'])
    # sample_pred = torch.argmax(model.recursive(sample_input, args['DEVICE']), dim=2).T # [seq, batch, features] -> [batch, seq]
    sample_pred = torch.argmax(model(sample_input, sample_target), dim=2).T
    # Transfer tensor to string
    sample_input_str = ''.join([dataset.indice_to_char[int(index)] for index in sample_input.squeeze(0)])
    sample_decinput_str = ''.join([dataset.indice_to_char[int(index)] for index in sample_target[:, :-1].squeeze(0)])
    sample_target_str = ''.join([dataset.indice_to_char[int(index)] for index in sample_target[:, 1:].squeeze(0)])
    sample_pred_str = ''.join([dataset.indice_to_char[int(index)] for index in sample_pred.squeeze(0)])



    acc = true_positive / samples
    loss_sum /= mask.shape[0]
    model.train()
    print('\n===============')
    print(f"Epoch: {epoch + 1} \n"
          f"Input: {sample_input_str} \n"
          f"Dec_input: {sample_decinput_str} \n"
          f"Ground truth: {sample_target_str} \n"
          f"Prediction:   {sample_pred_str} \n"
          f"Acc: {acc} \nLoss_avg: {loss_sum}")
    print('===============')
    return acc, loss_sum




if __name__ == "__main__":
    outputs_test = torch.randn(3, 300, 26)
    print(torch.argmax(outputs_test, dim=1).shape)
    print(torch.argmax(outputs_test, dim=1))
    from config import args

    l = lambda x: min(x ** -0.5, x * (args['WARMUP_STEP'] ** 1.5))
    print(0.003 * l(400))


