# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn

# =================================================================
# [新增] 1. RunningMeanStd: 标准的在线归一化层
# =================================================================
class RunningMeanStd(nn.Module):
    def __init__(self, insize, epsilon=1e-05):
        super(RunningMeanStd, self).__init__()
        self.insize = insize
        self.epsilon = epsilon
        # 注册 buffer，使其包含在 state_dict 中但不会被梯度更新
        self.register_buffer("running_mean", torch.zeros(insize))
        self.register_buffer("running_var", torch.ones(insize))
        self.register_buffer("count", torch.ones(1))

    def _update_mean_var(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.running_mean
        tot_count = self.count + batch_count

        new_mean = self.running_mean + delta * batch_count / tot_count
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        self.running_mean.copy_(new_mean)
        self.running_var.copy_(new_var)
        self.count.copy_(new_count)

    def forward(self, input):
        # 仅在训练模式下更新统计量
        if self.training:
            batch_mean = input.mean(dim=0)
            batch_var = input.var(dim=0, unbiased=False)
            
            # [修改] 将 int 强制转换为 Tensor，满足 JIT 的类型检查要求
            batch_count = torch.tensor(input.shape[0], dtype=torch.float, device=input.device)
            
            self._update_mean_var(batch_mean, batch_var, batch_count)

        # 归一化计算: (x - mean) / std
        return (input - self.running_mean) / torch.sqrt(self.running_var + self.epsilon)

class ActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        fixed_sigma=False,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()

        activation = get_activation(activation)  # nn.ELU()
        # [新增] 2. 初始化归一化层
        print(f"RunningMeanStd Init: Actor Dim={num_actor_obs}, Critic Dim={num_critic_obs}")
        self.obs_normalizer = RunningMeanStd(num_actor_obs)
        
        # 如果 Critic 观测空间不同，则独立归一化；否则共享
        if num_critic_obs != num_actor_obs:
            self.critic_obs_normalizer = RunningMeanStd(num_critic_obs)
        else:
            self.critic_obs_normalizer = self.obs_normalizer
        
        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):  # l=0,1,2  len(actor_hidden_dims)=3
            if l == len(actor_hidden_dims) - 1:   # l = 2,也就是最后一层！
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)  # 解包后按照顺序去组成神经网络

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))  # 这里的critic输出是1个
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise，用于探索
        self.fixed_sigma = fixed_sigma
        if fixed_sigma:
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions), requires_grad=False)
        else:
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))        
        self.distribution = None  # 存储这个概率分布的对象，生成一个带噪声的动作


        # disable args validation for speedup
        Normal.set_default_validate_args = False  # 这里是torch的高斯分布的类。将参数验证禁用。默认情况下，PyTorch 的分布类会验证输入参数的有效性（如标准差是否为正数等），这会增加一些计算开销

        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]  # 对线性层的权重进行正交化


    def reset(self, dones=None):
        pass  # 预留接口，用于重置RNN状态

    def forward(self):
        raise NotImplementedError  # 强制子类实现具体的前向传播过程
    
    @property
    def action_mean(self):
        return self.distribution.mean  # 动作分布的均值

    # @property
    # def action_std(self):
    #     return self.distribution.stddev  # 标准差
    @property
    def action_std(self):
        return self.std
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)  # 动作分布的熵，累加为一标量值（对动作维度的熵）
    #H(N(μ,σ2))=0.5*log(2πeσ^2) 这里是连续动作空间的高斯分布建模动作概率的熵的计算公式，均值和标准差是一个向量

    def update_distribution(self, observations):
        # [新增] 3. 训练时：输入先归一化，再进 Actor
        norm_obs = self.obs_normalizer(observations)
        mean = self.actor(norm_obs)  # 通过actor网络计算一个动作的均值
        self.distribution = Normal(mean, mean*0. + self.std)   # 固定标准差

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()  # 从分布中采样一个动作
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)  # 动作分布的对数概率，累加为一标量

    def act_inference(self, observations):
        # [新增] 4. 推理时：必须使用训练好的归一化参数
        norm_obs = self.obs_normalizer(observations)
        actions_mean = self.actor(norm_obs)
        return actions_mean  # 直接返回动作的均值（用于测试，确定性策略）

    def evaluate(self, critic_observations, **kwargs):
        # [新增] 5. Critic 也要归一化
        norm_critic_obs = self.critic_obs_normalizer(critic_observations)
        value = self.critic(norm_critic_obs)
        return value

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
