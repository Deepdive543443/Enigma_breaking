import torch
from torch.utils.tensorboard import SummaryWriter
import os, shutil
from time import gmtime, strftime
from tensorboard import program

class TensorboardLogger:
    def __init__(self, root = 'tensorboard'):
        #Logger
        self.root = root
        self.global_step = 0
        self.logger = {}

        print("Initial")

        # Make Path
        log_path = os.path.join(root, strftime("%a%d%b%Y%H%M%S", gmtime()))
        # os.makedirs(log_path)
        self.writer = SummaryWriter(log_path)

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

if __name__ == "__main__":
    # log = TensorboardLogger()
    # torch.save(log, 'log.th')

    log = torch.load('log.th')