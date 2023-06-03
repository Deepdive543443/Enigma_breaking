import torch
from torch.utils.tensorboard import SummaryWriter
import os, json
from model import Encoder, cp_2_key_model
from tensorboard import program

optim = torch.optim

class TensorboardLogger:
    def __init__(self, args):
        #Logger
        # self.global_step = 0
        self.logger = {}

        # Make Path
        # log_path = os.path.join(root, strftime("%a%d%b%Y%H%M%S", gmtime()))
        # os.makedirs(log_path)
        self.writer = SummaryWriter(args['LOG'])

        # Make a copy of config file
        with open(os.path.join(args['LOG'], 'config.json'), "w") as outfile:
            json.dump(args, outfile)

    def update(self, key, value):
        try:
            self.logger[key]['step'] += 1
            self.logger[key]['value'] = value
        except:
            self.logger[key] = {'value': value, 'step': 0}


        # self.global_step += 1
        for k, v in self.logger.items():
            self.writer.add_scalar(k, v['value'], global_step=v['step'])

    # def step(self):
    #     self.global_step += 1
    #     for k, v in self.logger.items():
    #         self.writer.add_scalar(k, v['value'], global_step=v['step'])

def launch_tensorboard(tracking_address):
    # https://stackoverflow.com/a/55708102
    # tb will run in background but it will
    # be stopped once the main process is stopped.
    try:
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', tracking_address, '--port', '8899'])
        url = tb.launch()
        if url.endswith("/"):
            url = url[:-1]

        return url
    except Exception:
        return None

def save_checkpoint(model, optimizer, args, info, filename):
    log_path = os.path.join(args['LOG'], filename)
    torch.save(
        {
            'weights': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'args': args,
            'info': info
        },
        log_path
    )

def load_checkpoint(filename):
    '''
    Load the checkpoint by its absolute path

    :param filename:
    :return:
    '''
    # Print the previous ckpt information
    ckpt = torch.load(filename)
    print(f"Loaded checkpoint start from:\n{ckpt['info']}")

    # Initial model and load the trained weights
    ckpt_args = ckpt['args']

    if ckpt_args['TYPE'] == 'Encoder':
        model = Encoder(ckpt_args)

    elif ckpt_args['TYPE'] == 'CP2K':
        model = cp_2_key_model

    model.load_state_dict(ckpt['weights'])
    model.to(ckpt_args['DEVICE'])

    optimizer = optim.Adam(model.parameters(), lr=ckpt_args['LEARNING_RATE'], betas=[ckpt_args['BETA1'], ckpt_args['BETA1']], eps=ckpt_args['EPS'])
    optimizer.load_state_dict(ckpt['optimizer'])
    return model, optimizer, ckpt_args

if __name__ == "__main__":
    # log = TensorboardLogger()
    # torch.save(log, 'log.th')

    log = torch.load('log.th')