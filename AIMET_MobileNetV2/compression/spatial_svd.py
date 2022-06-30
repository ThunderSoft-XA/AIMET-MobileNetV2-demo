import argparse
import logging
import os
from datetime import datetime
from decimal import Decimal
from typing import Tuple
from torchvision import models
import torch

# imports for AIMET
import aimet_common.defs
import aimet_torch.defs
from aimet_torch.compress import ModelCompressor

# imports for data pipelines
from common import image_net_config

from compression.data_pipeline import  ImageNetDataPipeline

from decimal import Decimal
from aimet_torch.defs import GreedySelectionParameters, SpatialSvdParameters
from aimet_common.defs import CompressionScheme, CostMetric
from aimet_torch.compress import ModelCompressor


def aimet_spatial_svd(model: torch.nn.Module,
                     evaluator: aimet_common.defs.EvalFunction) -> Tuple[torch.nn.Module,
                                                                         aimet_common.defs.CompressionStats]:
    greedy_params = GreedySelectionParameters(target_comp_ratio=Decimal(0.8),
                                            num_comp_ratio_candidates=3)
    modules_to_ignore = [model.features]
    auto_params = SpatialSvdParameters.AutoModeParams(greedy_select_params=greedy_params,
                                                    modules_to_ignore=modules_to_ignore)
    params = SpatialSvdParameters(mode=SpatialSvdParameters.Mode.auto, params=auto_params)

    eval_iterations = 1

    compress_scheme = CompressionScheme.spatial_svd

    cost_metric = CostMetric.mac

    results = ModelCompressor.compress_model(model=model,
                                            eval_callback=evaluator,
                                            eval_iterations=eval_iterations,
                                            input_shape=(1, 3, 224, 224),
                                            compress_scheme=compress_scheme,
                                            cost_metric=cost_metric,
                                            parameters=params)

    return results

def spatial_svd_example(model: torch.mode, config: argparse.Namespace, logger: logging.Logger):
    """
    1. Instantiate Data Pipeline for evaluation and training
    2. Load the pretrained MobileNetV2 model
    3. Calculate floating point accuracy
    4. Compression
        4.1. Compress the model using AIMET Weight SVD
        4.2. Log the statistics
        4.3. Save the compressed model
        4.4. Calculate and log the accuracy of compressed model
    5. Finetuning
        5.1 Finetune the compressed model
        5.2 Calculate and log the accuracy of compressed-finetuned model

    :param config: This argparse.Namespace config expects following parameters:
                   dataset_dir: Path to a directory containing ImageNet dataset.
                                This folder should conatin at least 2 subfolders:
                                'train': for training dataset and 'val': for validation dataset.
                   use_cuda: A boolean var to indicate to run the test on GPU.
                   logdir: Path to a directory for logging.
                   epochs: Number of epochs (type int) for finetuning.
                   learning_rate: A float type learning rate for model finetuning
                   learning_rate_schedule: A list of epoch indices for learning rate schedule used in finetuning. Check
                                           https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#MultiStepLR
                                           for more details.
    """

    # Instantiate Data Pipeline for evaluation and training
    data_pipeline = ImageNetDataPipeline(config)

    if config.use_cuda:
        model.to(torch.device('cuda'))
    model.eval()

    # Calculate floating point accuracy
    accuracy = data_pipeline.evaluate(model, use_cuda=config.use_cuda)
    logger.info("Original Model top-1 accuracy = %.2f", accuracy)

    # Compress the model using AIMET Weight SVD
    logger.info("Starting Weight SVD")
    compressed_model, stats = aimet_spatial_svd(model=model, evaluator=data_pipeline.evaluate)

    logger.info(stats)
    with open(os.path.join(config.logdir, 'log.txt'), "w") as outfile:
        outfile.write("%s\n\n" % (stats))

    # Calculate and log the accuracy of compressed model
    accuracy = data_pipeline.evaluate(compressed_model, use_cuda=config.use_cuda)
    logger.info("Compressed Model top-1 accuracy = %.2f", accuracy)

    logger.info("Weight SVD Complete")

    # Finetune the compressed model
    logger.info("Starting Model Finetuning")
    data_pipeline.finetune(compressed_model)

    # Calculate and log the accuracy of compressed-finetuned model
    accuracy = data_pipeline.evaluate(compressed_model, use_cuda=config.use_cuda)
    logger.info("After Weight SVD, Model top-1 accuracy = %.2f", accuracy)

    logger.info("Model Finetuning Complete")

    # Save the compressed model
    torch.save(compressed_model, os.path.join(config.logdir, 'compressed_model.pth'))
