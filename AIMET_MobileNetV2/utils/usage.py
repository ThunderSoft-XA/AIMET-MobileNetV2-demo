import argparse
import logging
import os
from datetime import datetime

def MobileNetV2_arguments():

	default_logdir = os.path.join("benchmark_output", "weight_svd_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

	parser = argparse.ArgumentParser(
        description='Apply Weight SVD on pretrained MobileNetV2 model and finetune it for ImageNet dataset')

	parser.add_argument('--dataset_dir', type=str,
						required=True,
						help="Path to a directory containing ImageNet dataset.\n\
								This folder should conatin at least 2 subfolders:\n\
								'train': for training dataset and 'val': for validation dataset")
	parser.add_argument('--use_cuda', action='store_true',
						help='Add this flag to run the test on GPU.')

	parser.add_argument('--logdir', type=str,
						default=default_logdir,
						help="Path to a directory for logging.\
								Default value is 'benchmark_output/weight_svd_<Y-m-d-H-M-S>'")

	parser.add_argument('--epochs', type=int,
						default=15,
						help="Number of epochs for finetuning.\n\
								Default is 15")
	parser.add_argument('--learning_rate', type=float,
						default=1e-2,
						help="A float type learning rate for model finetuning.\n\
								Default is 0.01")
	parser.add_argument('--learning_rate_schedule', type=list,
						default=[5, 10],
						help="A list of epoch indices for learning rate schedule used in finetuning.\n\
								Check https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#MultiStepLR for more details.\n\
								Default is [5, 10]")

	_config = parser.parse_args()
	return _config