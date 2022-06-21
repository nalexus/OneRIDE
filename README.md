# OneRIDE:
We propose a novel type of intrinsic reward based on RIDE and NoveID.

## Citation
If you use this code in your own work, please cite our paper:

## Installation

```
# create a new conda environment
conda create -n ride python=3.7
conda activate ride 

# install dependencies
git clone git@github.com:facebookresearch/impact-driven-exploration.git
cd impact-driven-exploration
pip install -r requirements.txt

# install MiniGrid
cd gym-minigrid
python setup.py install
```

## Train OneRIDE on MiniGrid MultiRoom N2-S4
```
cd impact-driven-exploration

OMP_NUM_THREADS=1 python main.py 

```

## Acknowledgements
Our algorithm is based on [RIDE](https://github.com/facebookresearch/impact-driven-exploration)

## License
This code is under the CC-BY-NC 4.0 (Attribution-NonCommercial 4.0 International) license.
