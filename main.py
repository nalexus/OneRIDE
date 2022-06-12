# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from src.arguments import parser
from src.arguments_bebold import parser as parser_bebold

from src.algos.torchbeast import train as train_vanilla 
from src.algos.count import train as train_count
from src.algos.curiosity import train as train_curiosity 
from src.algos.rnd import train as train_rnd
from src.algos.ride import train as train_ride
from src.algos.bebold import train as train_bebold
from src.algos.no_episodic_counts import train as train_no_episodic_counts
from src.algos.only_episodic_counts import train as train_only_episodic_counts

def main(flags):
    # if flags.model == 'vanilla':
    #     train_vanilla(flags)
    # elif flags.model == 'count':
    #     train_count(flags)
    # elif flags.model == 'curiosity':
    #     train_curiosity(flags)
    # elif flags.model == 'rnd':
    #     train_rnd(flags)
    # elif flags.model == 'ride':

    # train_ride(flags)

    train_bebold(flags)

    # elif flags.model == 'no-episodic-counts':
    #     train_no_episodic_counts(flags)
    # elif flags.model == 'only-episodic-counts':
    #     train_only_episodic_counts(flags)
    # else:
    #     raise NotImplementedError("This model has not been implemented. "\
    #     "The available options are: vanilla, count, curiosity, rnd, ride, \
    #     no-episodic-counts, and only-episodic-count.")

if __name__ == '__main__':
    # flags = parser.parse_args()
    flags = parser_bebold.parse_args()
    # flags.env = 'MiniGrid-KeyCorridorS3R1-v0'
    flags.env = 'MiniGrid-MultiRoom-N2-S4-v0'
    #flags.env = 'MiniGrid-ObstructedMaze-1Dl-v0'
    flags.total_frames = 1000000

    # # OneRIDE
    #
    # flags.intrinsic_reward_coef = 0.1
    # flags.entropy_cost = 0.0005
    # flags.num_actors = 3
    # flags.unroll_length = 100
    # flags.model = 'ride'
    # flags.disable_checkpoint = True
    # flags.num_threads = 1
    # flags.restore_back_time = 2
    # flags.learning_rate = 0.0001
    # # flags.adversarial_learning_rate=0.001
    # flags.max_grad_norm = 40
    # flags.forward_loss_coef = 10.0
    # flags.inverse_loss_coef = 0.1
    # # flags.adversarial_loss_coef = 10
    # flags.momentum=0
    # flags.epsilon=1e-05
    # flags.num_buffers=80
    # flags.rnd_loss_coef=0.1

    # NoveID (a.k.a BeBold)

    # flags.intrinsic_reward_coef = 0.05
    # flags.entropy_cost = 0.0005
    # flags.use_lstm = False
    flags.disable_checkpoint = True
    flags.num_threads = 1
    flags.num_actors = 3
    flags.unroll_length = 50

    main(flags)
