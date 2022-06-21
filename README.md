# OneRIDE:
We propose a novel type of intrinsic reward based on RIDE and NoveID. All results plotted in diploma are in folder results_for_plots.

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

If you want to change environment - make corresponding changes in main.py (uncomment necessary env). If you want to train ride or nove_id, uncomment train_ride or train_bebold. If you chose node_id, you may comment all configs for OneRIDE in main function and uncomment those for NoveID, also change parser respectively.

## Acknowledgements
Our implementation is based on [RIDE](https://github.com/facebookresearch/impact-driven-exploration).

## License
This code is under the CC-BY-NC 4.0 (Attribution-NonCommercial 4.0 International) license.
