## AIMET Setup Developer documentation

The main purpose of this project is to guide users to quickly master AIMET APIs, so that users can optimize their models more smoothly.

Due to the problem of my network and host resources, I installed the 1.21.0.

## The main development process:
* Prepare the data set and torch mobileNetV2
* Compression mobileNetV2  pretrained file

## Build aimet environment in conda
AIMET github project :
https://github.com/quic/aimet

Please copy the following files from the AIMET repository: 
File 1 from <link> to /AIMET/Utils


## Prepare the data set and torch mobileNetV2
The Torch version of  mobileNetV2 comes from the following open source projects:

https://github.com/tonylins/pytorch-mobilenet-v2
https://www.dropbox.com/s/47tyzpofuuyyv1b/mobilenetv2_1.0-f2a8633.pth.tar?dl=1

Follow the steps in the open source project to get the data set and network build scripts.

copy mobilenetv2_1.0-f2a8633.pth.tar model file to AIMET_MobileNetV2/pretrained

Download ImageNet dataset(Just extract it)
https://image-net.org/

## Compression mobileNetV2  pretrained file
Can execute python scripts directly:
```
$ conda activate aimet-env
$ python main.py --dataset_dir /home/HDD/dataset/ImageNet2012DataSets/
````

You can see all the evaluation results at benchmark_output and data.
