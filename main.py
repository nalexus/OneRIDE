# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from src.arguments import parser
from src.arguments_bebold import parser as parser_bebold
from src.algos.ride import train as train_ride
from src.algos.oneride import train as train_oneride
from src.algos.bebold import train as train_bebold

def main(flags):
    # train_ride(flags)
    # train_bebold(flags)
    train_oneride(flags)

if __name__ == '__main__':
    flags = parser.parse_args()
    #flags = parser_bebold.parse_args()
    # flags.env = 'MiniGrid-KeyCorridorS3R1-v0'
    #flags.env = 'MiniGrid-ObstructedMaze-1Dl-v0'
    flags.env = 'MiniGrid-MultiRoom-N2-S4-v0'
    flags.total_frames = 1000000

    # OneRIDE
    
    flags.intrinsic_reward_coef = 0.1
    flags.entropy_cost = 0.0005
    flags.num_actors = 3
    flags.unroll_length = 50
    flags.model = 'ride'
    flags.disable_checkpoint = True
    flags.num_threads = 1
    flags.restore_back_time = 2
    flags.learning_rate = 0.0001
    flags.max_grad_norm = 10


    # NoveID (a.k.a BeBold)
    
    # flags.disable_checkpoint = True
    # flags.num_threads = 1
    # flags.num_actors = 3

    main(flags)
