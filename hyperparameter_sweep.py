from src.arguments import parser
from src.algos.ride import train as train_ride
intrinsic_coef = [0.1,0.5,1]
unroll_length = [50,20,70]

flags = parser.parse_args()
flags.env = 'MiniGrid-MultiRoom-N7-S4-v0'
flags.total_frames = 1000000
# flags.intrinsic_reward_coef = 0.1
flags.entropy_cost = 0.0005
flags.num_actors = 3
# flags.unroll_length = 50
flags.model = 'ride'
flags.disable_checkpoint = True
flags.num_threads = 1

for coef in intrinsic_coef:
    flags.intrinsic_reward_coef = coef
    for length in unroll_length:
        flags.unroll_length = length
        train_ride(flags)