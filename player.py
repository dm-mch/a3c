#!/usr/bin/env python3

import numpy as np
from scipy import signal
import cv2
from threading import Thread
from time import sleep
from common import make_action
from collections import namedtuple


def process_frame(frame):
    """
    Preprocess obzervation
    """        
    frame = frame[34:34+160, :160]
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.reshape(frame, [42, 42, 1])
    return frame


def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

Batch = namedtuple("Batch", ["si", "a", "adv", "r", "terminal", "features"])

class Rollout(object):
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.r = 0.0
        self.terminal = False
        self.features = []

    def add(self, state, action, reward, value, terminal, features):
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal
        self.features += [features]

    def process(self, gamma):
        """
        compute returns and the advantage from rollout
        """
        batch_si = np.asarray(self.states)
        batch_a = np.asarray(self.actions)
        rewards = np.asarray(self.rewards)
        vpred_t = np.asarray(self.values + [self.r])

        rewards_plus_v = np.asarray(self.rewards + [self.r])
        batch_r = discount(rewards_plus_v, gamma)[:-1]
        delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
        # this formula for the advantage comes "Generalized Advantage Estimation":
        # https://arxiv.org/abs/1506.02438
        batch_adv = discount(delta_t, gamma * gamma)

        features = self.features[0]
        #print("bi", batch_si.shape, "a", batch_a.shape, 'r', batch_r.shape, 'adv', batch_adv.shape)
        return Batch(batch_si, batch_a, batch_adv, batch_r, self.terminal, features)


class Player(Thread):
    """
    Play episodes in separate thread
    Calc local gradient and apply it to remote target model

    """
    
    def __init__(self, id, env, model, config, render=False):
        """
        config: coord, limit_step, max_step, stat, gamma, actions
        """
        super(Player, self).__init__(target = Player.worker_loop, args = (self,))
        
        self.id = id
        self.name = "Player_" + str(self.id)
        self.env = env
        self.model = model
        self.model.name = self.name
        self.coord = config.coord
        self.limit = config.limit_step # total episode frames limit
        self.max_step = config.max_step # step for learn
        self.stat = config.stat[id]
        self.gamma = config.gamma
        self.actions = config.actions
        self.render = render


    def play_episode(self, mean_steps):
        ob = process_frame(self.env.reset())
        total_steps = 0
        total_rewards = 0
        done = False
        last_features = self.model.get_initial_features()
        # check total step for prevent "do nothing"  
        while not done and total_steps < (mean_steps * 100 or self.limit):
            self.model.sync_from_target()
            rollout = Rollout()

            for step in range(self.max_step):
                fetch =  self.model.predict(ob, *last_features)
                a, value, features = fetch[0][0],fetch[1],fetch[2:] 
                new_ob, r, done, info = self.env.step(self.actions[a])
                new_ob = process_frame(new_ob)

                # collect the experience
                rollout.add(ob, a, r, value, done, last_features)

                total_rewards += r
                total_steps += 1
                ob = new_ob
                last_features = features
                
                if self.render and step%10 == 0:
                    self.env.render()
 
                if done:
                    break
            if len(rollout.states) > 0:
                if not done:
                    rollout.r = self.model.predict_value(ob, *last_features)
                # get states, actions, discounted value for last steps 
                batch = rollout.process(self.gamma)

                # calc loacl and apply remote gradient
                # states, actions, values,  advs, lstm_states
                self.model.train(batch.si, batch.a, batch.r, batch.adv, batch.features)
        return total_steps, total_rewards
    
    
    @staticmethod
    def worker_loop(self):
        sleep(self.id)
        print("Player %d started!" % self.id)
        episode = 0
        rewards = []
        steps = []
        mean_steps = 0
        mean_rewards = 0
        try:
            while not self.coord.should_stop():
                step, reward = self.play_episode(mean_steps)
                rewards.append(reward)
                steps.append(step)
                #statistic
                mean_steps = np.mean(steps[-100:])
                mean_rewards = np.mean(rewards[-100:])
                #print(rewards[-100:])
                self.stat['mean_steps'] = mean_steps
                self.stat['mean_rewards'] = mean_rewards
                self.stat['max_rewards'] = np.max(rewards[-100:])
                self.stat['episode'] = episode
                episode += 1
        except Exception as e:
            self.coord.request_stop(e)
        finally:
            print("Player %d terminated. %d episodes played" % (self.id, episode))
