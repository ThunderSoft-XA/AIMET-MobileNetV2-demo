import os
import logging

import torch

from compression.weight_svd import weight_svd_example
from compression.spatial_svd import spatial_svd_example
from compression.channel_pruning import channel_pruning_example

from utils.usage import MobileNetV2_arguments

from model.MobileNetV2 import mobilenet_v2



logger = logging.getLogger('TorchWeightSVD')
formatter = logging.Formatter('%(asctime)s : %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(format=formatter)


if __name__ == '__main__':

    _config = MobileNetV2_arguments()
    model = mobilenet_v2(pretrained=True)

    os.makedirs(_config.logdir, exist_ok=True)

    fileHandler = logging.FileHandler(os.path.join(_config.logdir, "test.log"))
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    if _config.use_cuda and not torch.cuda.is_available():
        logger.error('use_cuda is selected but no cuda device found.')
        raise RuntimeError("Found no CUDA Device while use_cuda is selected")

    weight_svd_example(model, _config, logger)
