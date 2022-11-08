# Multi-Sensor
This package is built based on public framework [MMF](https://github.com/facebookresearch/mmf) for vision and language multimodal research from Facebook AI Research. Follow the MMF installation and move all folders in this repository to the root folder of MMF.

## Installation

Follow installation instructions in the [documentation](https://mmf.sh/docs/).

## Documentation

Learn more about MMF [here](https://mmf.sh/docs).


## Usage
```Shell
# To train the multi sensor model on the physionet sepsis training set:
mmf_run dataset=physionet_sepsis \
        model=multi_sensor_pt \
        config=projects/multi_sensor_pt/configs/physionet_sepsis/defaults.yaml
```
