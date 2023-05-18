from torch.utils.tensorboard import SummaryWriter
import os, shutil, torch


class TensorboardLogger:
    def __init__(self, root = 'tensorboard'):
        #Logger
        self.root = root
        self.global_step = 0
        self.logger = {}

        #Make logging file
        num_logs = len(os.listdir(self.root))
        os.makedirs(num_logs)
        self.writer = SummaryWriter(os.path.join(self.root, num_logs))

    def update(self, key, value):
        try:
            self.logger[key]['step'] += 1
            self.logger[key]['value'] = value
        except:
            self.logger[key] = {'value': value, 'step': 0}

    def step(self):
        self.global_step += 1
        for k, v in self.logger.items():
            self.writer.add_scalar(k, v['value'], global_step=v['step'])


def save_checkpoint(model, optimizer, args, logger, type_ckpt):
    # Make an folder checkpoint if we don't have it
    os.mkdir('ckpt')
    num_ckpt = len(os.listdir('ckpt'))
    name_ckpt = str(num_ckpt) + f"_{type_ckpt}"
    
    # Saving the weight and configuration of model, and the status of Adam optimizer
    torch.save(
        {
            'weights':model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'args': args,
            'logger': logger
        },
        os.path.join('ckpt', name_ckpt)
    )

def load_checkpoint(model, optimizer, path):
    ckpt = torch.load(path)
    
    # Loading weight and stat for model and optimizer
    model.load_state_dict(ckpt['weights'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    args = ckpt['args']
    logger = ckpt['logger']
    return model, optimizer, args