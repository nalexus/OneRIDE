import src.models as models
import torch
from src.arguments import parser
from src.utils import get_batch, log, create_env, create_buffers, act

MinigridStateEmbeddingNet = models.MinigridStateEmbeddingNet
MinigridPolicyNet = models.MinigridPolicyNet

if __name__ == '__main__':
    flags = parser.parse_args()

    flags.device = torch.device('cpu')

    env = create_env(flags)
    model = MinigridPolicyNet(env.observation_space.shape, env.action_space.n)

    breakpoint()
    state = model.initial_state(batch_size=1)
    # agent_state = model.initial_state(batch_size=1)
    # batch, agent_state = get_batch(free_queue, full_queue, buffers,
    #                                initial_agent_state_buffers, flags, timings)
    state_embedding_model = MinigridStateEmbeddingNet(env.observation_space.shape)\
                    .to(device='cpu')

    state_emb = state_embedding_model(state[0]) \
        .reshape(flags.unroll_length, flags.batch_size, 128)

