from argparse import ArgumentParser


def boolean(s: str) -> bool:
    """Convert string to boolean"""
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


base_parser = ArgumentParser(add_help=False)

# General script settings
base_parser.add_argument('--seed', type=int, default=42, help='Random seed')
base_parser.add_argument('--debug', action='store_true', help='Debug mode')
base_parser.add_argument('--no_train', action='store_false', dest='train',
                         help='Disable training')
base_parser.add_argument('--resume_path', type=str,
                         help='W&B path to checkpoint to resume training from'
                         ' in the format <entity>/<project>/<run_id>/<model_name>')

# Data settings
base_parser.add_argument('--train_dataset', type=str,
                         default='camcan', help='Training dataset name')  # camcan
base_parser.add_argument('--test_dataset', type=str, default='brats',
                         help='Test dataset name')  # brats
base_parser.add_argument('--val_split', type=float,
                         default=0.1, help='Validation fraction')
base_parser.add_argument('--sequence', type=str,
                         default='t1', help='MRI sequence')
base_parser.add_argument('--num_workers', type=int,
                         default=4, help='Number of workers')
base_parser.add_argument('--image_size', type=int,
                         default=128, help='Image size')
base_parser.add_argument('--slice_range', type=int,
                         nargs='+', default=(55, 135), help='Slice range')
base_parser.add_argument('--normalize', type=boolean,
                         default=False, help='Normalize images between 0 and 1')
base_parser.add_argument('--equalize_histogram', type=boolean,
                         default=True, help='Equalize histogram')
base_parser.add_argument('--anomaly_size', type=int,
                         nargs='+', default=(10, 20),
                         help='size of artificial anomalies')
base_parser.add_argument('--anomaly_name', type=str, default='source_deformation',
                         choices=['source_deformation',
                                  'sink_deformation',
                                  'intensity_anomaly'])

# Logging settings
base_parser.add_argument('--val_frequency', type=int,
                         default=200, help='Validation frequency')
base_parser.add_argument('--val_steps', type=int,
                         default=50, help='Steps per validation')
base_parser.add_argument('--log_frequency', type=int,
                         default=50, help='Logging frequency')
base_parser.add_argument('--save_frequency', type=int, default=200,
                         help='Model checkpointing frequency')
base_parser.add_argument('--num_images_log', type=int,
                         default=10, help='Number of images to log')

# Hyperparameters
base_parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
base_parser.add_argument('--weight_decay', type=float,
                         default=0.0, help='Weight decay')
base_parser.add_argument('--max_steps', type=int,
                         default=10000, help='Number of training steps')
base_parser.add_argument('--batch_size', type=int,
                         default=32, help='Batch size')

# Model settings
base_parser.add_argument('--model', type=str, default='FeatureReconstructor',
                         help='Model name')
base_parser.add_argument('--loss_fn', type=str, default='ssim',
                         choices=['ssim', 'mse'], help='Loss function')
base_parser.add_argument('--hidden_dims', type=int, nargs='+',
                         default=[100, 150, 200, 300],
                         help='Autoencoder hidden dimensions')
base_parser.add_argument('--dropout', type=float,
                         default=0.1, help='Dropout rate')
base_parser.add_argument('--extractor_cnn_layers',
                         type=str, nargs='+',
                         default=['layer0', 'layer1', 'layer2'])
base_parser.add_argument('--keep_feature_prop', type=float,
                         default=1.0,
                         help='Proportion of ResNet features to keep')
base_parser.add_argument('--random_extractor', action="store_true",
                         help="Use untrained extractor")
