# Reinforcement_Learning_in_Action
强化学习快速上手：编写自定义通用gym环境类+主流开源强化学习框架调用

前言

相信很多同学接触强化学习都是从使用OpenAI提供的gym示例开始，跟着讲义一步步开发自己的算法程序。这个过程虽然能够帮助我们熟悉强化学习的理论基础，却有着陡峭的学习曲线，需要耗费大量的时间精力。对于那些仅仅是想借助强化学习解决问题的同学来说，这个过程显得过于漫长了。熟悉机器学习那一套的同学都知道，通常我们仅仅需要将数据组织成合适的格式，调用scikit-learn中的算法就可以了。类似地，由于强化学习的算法大多已经有成熟的开源框架可用，我们仅仅需要定义具体的研究对象即可。

预备

强化学习基本知识：智能体agent与环境environment、状态states、动作actions、回报rewards等等，网上都有相关教程，不再赘述。
gym安装：openai/gym 注意，直接调用pip install gym只会得到最小安装。如果需要使用完整安装模式，调用pip install gym[all]。
主流开源强化学习框架推荐如下。以下只有前三个原生支持gym的环境，其余的框架只能自行按照各自的格式编写环境，不能做到通用。并且前三者提供的强化学习算法较为全面，PyBrain提供了较基础的如Q-learning、Sarsa、Natural AC算法。Tensorlayer基于tensorflow开发，提供了更为基本的api。由于我始终未能安装上coach……本文将以前两个为例。
OpenAI提供的Baselines：openai/baselines 3.2k stars
Tensorforce：reinforceio/tensorforce 1.2k stars
Intel提供的Coach：NervanaSystems/coach 0.6k stars，pip安装可以使用pip install rl-coach，而不是pip install coach
PyBrain：pybrain/pybrain 2.5k stars
Tensorlayer：tensorlayer/tensorlayer 3.1k stars
编写自定义环境

参考gym中经典的CartPole环境代码CartPole.py，我们逐步构建自定义环境。

我们将要创建的是一个在二维水平面上移动的小车。该二维区域长宽各20单位。区域中心为坐标原点，也是小车要到达的目的地。

动作：小车每次只能选择不动或向四周移动一个单位。

状态：小车的横纵坐标。

奖励：小车到达目的地周围有+10的奖励；每移动一个单位有-0.1的奖励，这表示我们希望小车尽量以较少的时间（移动次数）到达终点；小车移动到区域外的奖励为-50。

首先新建一个Car2D.py的文件，需要import的包如下：

import gym
from gym import spaces
import numpy as np
我们声明一个Car2DEnv的类，它是gym.Env的子类。它一般包括的内容如下：

class Car2DEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }
     
    def __init__(self):
        self.action_space = None
        self.observation_space = None
        pass
    
    def step(self, action):
        return self.state, reward, done, {}
    
    def reset(self):
        return self.state
        
    def render(self, mode='human'):
        return None
        
    def close(self):
        return None
必须实现的内容：

__init__()：将会初始化动作空间与状态空间，便于强化学习算法在给定的状态空间中搜索合适的动作。gym提供了spaces方法，详细内容可以help查看。；

step()：用于编写智能体与环境交互的逻辑，它接受action的输入，给出下一时刻的状态、当前动作的回报、是否结束当前episode及调试信息。输入action由__init__()函数中的动作空间给定。我们规定当action为0表示小车不动，当action为1，2，3，4时分别是向上、下、左、右各移动一个单位。据此可以写出小车坐标的更新逻辑；

reset()：用于在每轮开始之前重置智能体的状态。

不是必须实现的但有助于调试算法的内容：

metadata、render()、close()是与图像显示有关的，我们不涉及这一部分，感兴趣的同学可以自行编写相关内容。

完整的Car2D.py代码如下

# -*- coding: utf-8 -*-

import gym
from gym import spaces
import numpy as np

class Car2DEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }
     
    def __init__(self):
        self.xth = 0
        self.target_x = 0
        self.target_y = 0
        self.L = 10
        self.action_space = spaces.Discrete(5) # 0, 1, 2，3，4: 不动，上下左右
        self.observation_space = spaces.Box(np.array([self.L, self.L]), np.array([self.L, self.L]))
        self.state = None
    
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        x, y = self.state
        if action == 0:
            x = x
            y = y
        if action == 1:
            x = x
            y = y + 1
        if action == 2:
            x = x
            y = y - 1
        if action == 3:
            x = x - 1
            y = y
        if action == 4:
            x = x + 1
            y = y
        self.state = np.array([x, y])
        self.counts += 1
            
        done = (np.abs(x)+np.abs(y) <= 1) or (np.abs(x)+np.abs(y) >= 2*self.L+1)
        done = bool(done)
        
        if not done:
            reward = -0.1
        else:
            if np.abs(x)+np.abs(y) <= 1:
                reward = 10
            else:
                reward = -50
            
        return self.state, reward, done, {}
    
    def reset(self):
        self.state = np.ceil(np.random.rand(2)*2*self.L)-self.L
        self.counts = 0
        return self.state
        
    def render(self, mode='human'):
        return None
        
    def close(self):
        return None
        
if __name__ == '__main__':
    env = Car2DEnv()
    env.reset()
    env.step(env.action_space.sample())
    print(env.state)
    env.step(env.action_space.sample())
    print(env.state)
调用强化学习框架

Baselines

由于Baselines与gym都是一家开发的，代码上基本无缝衔接，直接import我们声明的这个Car2DEnv就可以使用了。以DQN为例，以下是训练+测试的代码：

from baselines import deepq
from Car2D import Car2DEnv

env = Car2DEnv()

model = deepq.models.mlp([32, 16], layer_norm=True)
act = deepq.learn(
    env,
    q_func=model,
    lr=0.01,
    max_timesteps=10000,
    print_freq=1,
    checkpoint_freq=1000
)

print('Finish!')
#act.save("mountaincar_model.pkl")

#act = deepq.load("mountaincar_model.pkl")
while True:
    obs, done = env.reset(), False
    episode_reward = 0
    while not done:
        env.render()
        obs, reward, done, _ = env.step(act(obs[None])[0])
        episode_reward += reward
    print([episode_reward, env.counts])
训练部分：调用了多层感知机mlp作为我们的简化深度Q网络。deepq.learn()中lr表示学习率，max_timesteps表示本次训练的总时间步steps达到10000步后结束（是的，你没看错！这不是episode而是时间步steps），print_freq表示每隔多少episode打印一次统计信息，checkpoint_freq表示每隔多少steps保存一次模型。最终将选择已保存的模型中平均回报最高的模型，赋给变量act。act可以save为pkl文件，方便下次load。

测试部分：act接受当前状态obs后给出action，将其传给环境的step()函数，得到下一时间步的状态、回报、是否结束。我们在Car2DEnv中有一个变量counts记录了每轮从开始到结束的时间步数，表示小车需要的时间。循环打印每轮的总回报和时间步数。如果总回报为正且时间步数较少，则表明我们的算法取得了较好的效果。



Tensorforce

Tensorforce通过调用方法OpenAIGym将已注册的gym环境导入。框架设计与Baselines略有不同。以PPO算法为例，直接看代码：

# -*- coding: utf-8 -*-

import numpy as np
from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym
env = OpenAIGym('Car2D-v0', visualize=False)

# Network as list of layers
network_spec = [
    dict(type='dense', size=32, activation='tanh'),
    dict(type='dense', size=32, activation='tanh')
]

agent = PPOAgent(
    states_spec=env.states,
    actions_spec=env.actions,
    network_spec=network_spec,
    batch_size=4096,
    # BatchAgent
    keep_last_timestep=True,
    # PPOAgent
    step_optimizer=dict(
        type='adam',
        learning_rate=1e-3
    ),
    optimization_steps=10,
    # Model
    scope='ppo',
    discount=0.99,
    # DistributionModel
    distributions_spec=None,
    entropy_regularization=0.01,
    # PGModel
    baseline_mode=None,
    baseline=None,
    baseline_optimizer=None,
    gae_lambda=None,
    # PGLRModel
    likelihood_ratio_clipping=0.2,
    summary_spec=None,
    distributed_spec=None
)

# Create the runner
runner = Runner(agent=agent, environment=env)

# Callback function printing episode statistics
def episode_finished(r):
    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                                 reward=r.episode_rewards[-1]))
    return True

# Start learning
runner.run(episodes=1000, max_episode_timesteps=200, episode_finished=episode_finished)

# Print statistics
print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    ep=runner.episode,
    ar=np.mean(runner.episode_rewards[-100:]))
)
    
while True:
    agent.reset()
    state, done = env.reset(), False
    episode_reward = 0
    while not done:
        action = agent.act(state, deterministic = True)
        state, done, reward = env.execute(action)
        agent.observe(done, reward)
        episode_reward += reward
    print([episode_reward])
为了让代码能够顺利运行，我们需要额外配置一些东西：

1. 注册自定义gym环境

首先找到gym环境的文件夹位置。我使用的是anaconda，路径是D:\Anaconda\Lib\site-packages\gym\gym\envs。新建一个文件夹user并进入。将刚才我们编写的Car2D.py放进去。并增加入口文件__init__.py，内容为：

from gym.envs.user.Car2D import Car2DEnv
回到D:\Anaconda\Lib\site-packages\gym\gym\envs。修改入口文件__init__.py（不放心的可以备份原文件），在其中增加内容：

# User
# ----------------------------------------

register(
    id='Car2D-v0',
    entry_point='gym.envs.user:Car2DEnv',
    max_episode_steps=100,
    reward_threshold=10.0,
)
2. 修改 D:\Anaconda\Lib\site-packages\tensorforce\execution\runner.py，将该文件最后两行注释掉，如下（不放心的同学也可以备份一下）

        #self.agent.close()
        #self.environment.close()
原因是这将导致agent和environment在训练完成后被清空，使得测试部分无法进行。

配置好这两项内容后就可以愉快地运行代码啦！Tensorforce中使用runner()进行训练，对相关参数有兴趣的同学可以阅读官方文档相关内容。代码中agent.observe()仅仅是用于更新agent.episode信息，不会更新训练好的模型参数。需要说明的是，由于tensorforce通过导入的形式调用自定义环境，我们自定义的内容如env.counts的正确调用形式是env.gym.env.counts。

后记

由于本人目前正在学习强化学习ing，难免有疏漏和错误。欢迎大家提出批评意见，共同进步！
