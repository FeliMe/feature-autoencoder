from argparse import ArgumentParser
from time import perf_counter

import wandb

from fae.data import datasets
from fae.data.data_utils import train_val_split
from fae.models import models
from fae.utils.utils import seed_everything


""""""""""""""""""""""""""""""""""" Config """""""""""""""""""""""""""""""""""
parser = ArgumentParser()

# General script settings
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--debug', action='store_true', help='Debug mode')
parser.add_argument('--resume', action='store_true', help='Debug mode')  # TODO: implement

# Model settings
parser.add_argument('--model', type=str, default='FeatureAutoencoder', help='Model name')
parser.add_argument('--c_latent', type=int, default=128, help='Latent channels')
parser.add_argument('--keep_feature_prop', type=float, default=1.0, help='Proportion of ResNet features to keep')

# Data settings
parser.add_argument('--train_dataset', type=str, default='camcan', help='Training dataset name')
parser.add_argument('--test_dataset', type=str, default='mslub', help='Test dataset name')
parser.add_argument('--val_split', type=float, default=0.1, help='Validation fraction')
parser.add_argument('--sequence', type=str, default='t1', help='MRI sequence')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
parser.add_argument('--image_size', type=int, default=224, help='Image size')
parser.add_argument('--slice_range', type=int, nargs='+', default=(55, 135), help='Slice range')
parser.add_argument('--normalize', action='store_true', help='Normalize images between 0 and 1')
parser.add_argument('--equalize_histogram', action='store_true', help='Equalize histogram')
parser.add_argument('--anomaly_size', type=int, nargs='+', default=(55, 135), help='Anomaly size')

# Hyperparameters
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--max_steps', type=int, default=10000, help='Number of training steps')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')

# Logging settings
parser.add_argument('--val_frequency', type=int, default=100, help='Validation frequency')

args = parser.parse_args()

wandb.init(project="feature_autoencoder", entity="felix-meissen", config=args,
           mode="disabled" if args.debug else "online")
config = wandb.config


""""""""""""""""""""""""""""""" Reproducability """""""""""""""""""""""""""""""
seed_everything(config.seed)


""""""""""""""""""""""""""""""""" Init model """""""""""""""""""""""""""""""""


def get_model(config):
    if config.model in models.__dict__:
        model_cls = models.__dict__[config.model]
    else:
        raise ValueError(f'Model {config.model} not found')

    return model_cls(config)


model = get_model(config)


""""""""""""""""""""""""""""""""" Load data """""""""""""""""""""""""""""""""


t_load_data_start = perf_counter()
train_loader, val_loader, test_loader = datasets.get_dataloaders(config)
print(f'Loaded {config.train_dataset} and {config.test_dataset} in '
      f'{perf_counter() - t_load_data_start:.2f}s')
import IPython; IPython.embed(); exit(1)


""""""""""""""""""""""""""""""""""" Train """""""""""""""""""""""""""""""""""


def train(model, trainloader, valloader, testloader, config):
    pass
