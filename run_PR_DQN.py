from RL_brain import DQNPrioritizedReplay
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from importlib import reload
import laser_hockey_env as lh
import time

t1 = time.time()

reload(lh)
playerComputer = lh.BasicOpponent()

env = lh.LaserHockeyEnv(mode=0)
MEMORY_SIZE = 100000
Ep_max = 10000
Step_max = 500

sess = tf.Session()

with tf.variable_scope('DQN_with_prioritized_replay'):
    RL_prio = DQNPrioritizedReplay(
        n_actions=8, n_features=16, memory_size=MEMORY_SIZE,
        e_greedy_increment=None, sess=sess, prioritized=True, output_graph=True,
    )
sess.run(tf.global_variables_initializer())


def train(RL):
    global GLOBAL_RUNNING_R
    total_steps = 0
    steps = []
    episodes = []
    cost_his = []
    for i_episode in range(Ep_max):
        ep_reward = 0
        observation = env.reset()
        epsilon = 1
        cost = 0
        for j in range(Step_max):
            # env.render()
            max_action_repeat_times = 0

            if max_action_repeat_times % 4 == 0:
                action = RL.choose_action(observation)
                action_repeat = action
            else:
                action = action_repeat
                max_action_repeat_times += 1
            a = env.discrete_to_continous_action(action)

            # if i_episode < 3000:
            a_opp = playerComputer.act(env.obs_agent_two())
            # else:
            #     a_opp_action = RL.choose_action(env.obs_agent_two())
            #     a_opp = - env.discrete_to_continous_action(a_opp_action)

            a_6 = np.hstack([a, a_opp])

            observation_, reward, done, info = env.step(a_6)
            reward_touch_puck = info['reward_touch_puck']
            hit = info["hit"]

            p1t1, v1t1 = observation[:2], np.clip(observation[3:5], -30, 30)
            ppt1, vpt1 = observation[12:14], np.clip(observation[14:], -30, 30)
            p1t2, v1t2 = observation_[:2], np.clip(observation_[3:5], -30, 30)
            ppt2, vpt2 = observation_[12:14], np.clip(observation_[14:], -30, 30)
            deltay_t1 = abs(p1t1[1] - ppt1[1])
            deltay_t2 = abs(p1t2[1] - ppt2[1])
            deltax_t1 = abs(p1t1[0] - ppt1[0])
            deltax_t2 = abs(p1t2[0] - ppt2[0])

            if reward_touch_puck > 0:  # for hit the puck
                if p1t2[0] < ppt2[0]:  # hit the ball from left side
                    reward += 1

                elif ppt2[0] < p1t2[0]:  # hit the ball from right side
                    reward -= 1

            if deltay_t1 > deltay_t2:  # for control y direction
                if deltay_t2 < 0.01:  # move closer in y direction is good
                    reward += 1
                else:
                    reward += (deltay_t1 / deltay_t2 - 1)
            else:
                if deltay_t1 < 0.01:  # move faraway in y direction is bad
                    reward -= 1
                else:
                    reward -= (deltay_t2 / deltay_t1 - 1)

            if ppt1[0] < 0:  # for control x direction
                if vpt1[0] == 0:
                    reward += v1t2[0] * 0.01  # hit the ball more bravely
            if ppt1[0] > 0:
                reward -= v1t2[0] * 0.005  # move back to keep the goal

            reward = np.clip(reward, -2, 2)
            ep_reward += reward

            RL.store_transition(observation, action, reward, done, observation_)

            if total_steps > MEMORY_SIZE and total_steps % 10 == 0:
                epsilon, cost = RL.learn()
                cost_his.append(cost)

            if done:
                steps.append(total_steps)
                episodes.append(i_episode)
                break

            observation = observation_
            total_steps += 1
        print('episode ', i_episode, 'total_steps: ', total_steps, 'done:', done, 'hit:', hit, 'eplison:', epsilon, )

        # record reward changes, plot later
        if len(GLOBAL_RUNNING_R) == 0:
            GLOBAL_RUNNING_R.append(ep_reward)
        else:
            GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1] * 0.9 + ep_reward * 0.1)
        if i_episode % 100 == 0:
            plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
            plt.xlabel('Episode')
            plt.ylabel('Moving reward')
            plt.savefig('PR_DQN_reward.png')
            plt.close()
            plt.plot(range(len(cost_his)), cost_his)
            plt.xlabel('learning step')
            plt.ylabel('cost history')
            plt.savefig('PR_DQN_cost.png')
            plt.close()
    return np.vstack((episodes, steps)), RL.q


GLOBAL_RUNNING_R = []
his_prio, Qvalue = train(RL_prio)

plt.plot(np.arange(len(Qvalue)), Qvalue)
plt.xlabel('step')
plt.ylabel('Moving Qvalue')
plt.savefig('PR_DQN_Qvalue.png')
plt.close()

RL_prio.save()

print('Running time: ', time.time() - t1)

win = 0
round = 10
for i in range(round):
    s = env.reset()
    for t in range(600):
        env.render()
        a = RL_prio.choose_action(s)
        a = env.discrete_to_continous_action(a)
        a_opp = playerComputer.act(env.obs_agent_two())
        a_6 = np.hstack([a, a_opp])
        s, _, done, info = env.step(a_6)
        if done:
            winner = info['winner']
            if winner == 1:
                win += 1
            break
print('win rate in test:', win / (round))
