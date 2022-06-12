# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn 
from torch.nn import functional as F
from copy import deepcopy
import numpy as np 


def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(torch.mean(advantages**2, dim=1))


def compute_entropy_loss(logits):
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    entropy_per_timestep = torch.sum(-policy * log_policy, dim=-1)
    return -torch.sum(torch.mean(entropy_per_timestep, dim=1))


def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction='none')
    cross_entropy = cross_entropy.view_as(advantages)
    advantages.requires_grad = False
    policy_gradient_loss_per_timestep = cross_entropy * advantages
    return torch.sum(torch.mean(policy_gradient_loss_per_timestep, dim=1))

def compute_adversarial_loss(logits, actions):
    adversarial_loss_func = nn.CrossEntropyLoss(reduction='none')
    loss = torch.mean(adversarial_loss_func(logits.permute(0, 2, 1),
                          actions.long()))
    return loss
    # cross_entropy = F.nll_loss(
    #     F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
    #     target=torch.flatten(actions, 0, 1),
    #     reduction='none')
    # cross_entropy = cross_entropy.view_as(advantages)
    # advantages.requires_grad = False
    # policy_gradient_loss_per_timestep = cross_entropy * advantages
    # return torch.sum(torch.mean(cross_entropy, dim=1))


def compute_forward_dynamics_loss(pred_next_emb, next_emb):
    forward_dynamics_loss = torch.norm(pred_next_emb - next_emb, dim=2, p=2)
    return torch.sum(torch.mean(forward_dynamics_loss, dim=1))


def compute_inverse_dynamics_loss(pred_actions, true_actions):
    inverse_dynamics_loss = F.nll_loss(
        F.log_softmax(torch.flatten(pred_actions, 0, 1), dim=-1), 
        target=torch.flatten(true_actions, 0, 1), 
        reduction='none')
    inverse_dynamics_loss = inverse_dynamics_loss.view_as(true_actions)
    return torch.sum(torch.mean(inverse_dynamics_loss, dim=1))

def indicator_transform_count_rewards(count_rewards):
    return torch.tensor([[float(j == 1) for j in i] for i in count_rewards])
    # return torch.tensor([[1 if j == 1 else -np.exp(-j) for j in i] for i in count_rewards])

def penalize_unit_time_lapse(time,count_rewards):
    num_rows = time.shape[0]
    num_cols = time.shape[1]
    return torch.tensor([[0 if (time[i][j] != 1) else -np.log(count_rewards[i][j])
                   for j in range(num_cols)] for i in range(num_rows)])

def restore_intrinsic_reward_after_time_lapse(restore_back_time,time,count_rewards):
    num_rows = time.shape[0]
    num_cols = time.shape[1]
    return torch.tensor([[1 if time[i][j] >= restore_back_time else count_rewards[i][j] for j in range(num_cols)] for i in range(num_rows)])

