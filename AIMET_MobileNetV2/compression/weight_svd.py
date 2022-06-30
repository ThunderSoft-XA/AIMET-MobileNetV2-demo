# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2021, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

"""
This file demonstrates the use of compression using AIMET weight SVD
technique followed by fine tuning.
"""

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

from compression.data_pipeline import ImageNetDataPipeline

def aimet_weight_svd(model: torch.nn.Module,
                     evaluator: aimet_common.defs.EvalFunction) -> Tuple[torch.nn.Module,
                                                                         aimet_common.defs.CompressionStats]:
    """
    Compresses the model using AIMET's Weight SVD auto mode compression scheme.

    :param model: The model to compress.
    :param evaluator: Evaluator used during compression.
    :return: A tuple of compressed model and its statistics
    """

    # Please refer to the API documentation for more details.

    # Desired target compression ratio using Weight SVD
    # This value denotes the desired compression % of the original model.
    # To compress the model to 20% of original model, use 0.2. This would
    # compress the model by 80%.
    # We are compressing the model by 50% here.
    target_comp_ratio = Decimal(0.5)

    # Number of compression ratio used by the API at each layer
    # API will evaluate 0.1, 0.2, ..., 0.9, 1.0 ratio (total 10 candidates)
    # at each layer
    num_comp_ratio_candidates = 10

    # Creating Greedy selection parameters:
    greedy_params = aimet_torch.defs.GreedySelectionParameters(target_comp_ratio=target_comp_ratio,
                                                               num_comp_ratio_candidates=num_comp_ratio_candidates)

    # Selecting 'greedy' for rank select scheme:
    rank_select_scheme = aimet_common.defs.RankSelectScheme.greedy

    # Ignoring first convolutional layer of the model for compression
    modules_to_ignore = [model.features]

    # Creating Auto mode Parameters:
    auto_params = aimet_torch.defs.WeightSvdParameters.AutoModeParams(rank_select_scheme=rank_select_scheme,
                                                                      select_params=greedy_params,
                                                                      modules_to_ignore=modules_to_ignore)

    # Creating Weight SVD parameters with Auto Mode:
    params = aimet_torch.defs.WeightSvdParameters(aimet_torch.defs.WeightSvdParameters.Mode.auto,
                                                  auto_params)

    # Scheme is Weight SVD:
    scheme = aimet_common.defs.CompressionScheme.weight_svd

    # Cost metric is MAC, it can be MAC or Memory
    cost_metric = aimet_common.defs.CostMetric.mac

    # Input image shape
    image_shape = (1, image_net_config.dataset['image_channels'],
                   image_net_config.dataset['image_width'], image_net_config.dataset['image_height'])

    # Calling model compression using Weight SVD:
    # Here evaluator is passed which is used by the API to evaluate the
    # accuracy for various compression ratio of each layer. To speed up
    # the process, only 10 batches of data is being used inside evaluator
    # (by passing eval_iterations=10) instead of running evaluation on
    # complete dataset.
    results = ModelCompressor.compress_model(model=model,
                                             eval_callback=evaluator,
                                             eval_iterations=10,
                                             input_shape=image_shape,
                                             compress_scheme=scheme,
                                             cost_metric=cost_metric,
                                             parameters=params)

    return results


def weight_svd_example(model: torch.mode, config: argparse.Namespace, logger: logging.Logger):
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
    compressed_model, stats = aimet_weight_svd(model=model, evaluator=data_pipeline.evaluate)

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