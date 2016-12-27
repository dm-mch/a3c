#!/usr/bin/env python3

import numpy as np
from threading import Thread
from time import sleep
from common import make_action

# import cProfile

# def profile(fn):
#     def profiled_fn(*args, **kwargs):
#         # name for profile dump
#         fpath = fn.__name__ + '.profile'
#         prof = cProfile.Profile()
#         ret = prof.runcall(fn, *args, **kwargs)
#         prof.dump_stats(fpath)
#         return ret
#     return profiled_fn


class Player(Thread):
    """
    Play episodes in separate thread
    Calc local gradient and apply it to remote target model

    """
    
    def __init__(self, id, env, model, config):
        """
        config: coord, limit_step, max_step
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
        self.render = config.render


    def play_episode(self, mean_steps):
        ob = self.env.reset()
        total_steps = 0
        total_rewards = 0
        done = False
        # check total step for prevent "do nothing"  
        while not done and total_steps < (mean_steps * 100 or self.limit):
            self.model.sync_from_target()
            for step in range(self.max_step):
                a = make_action(self.model.predict_policy(ob))
                ob, r, done, info = self.env.step(a)
                total_rewards += r
                total_steps += 1

                if self.render and step%10 == 0:
                    self.env.render()
 
                if done:
                    break
            predicted = 0
            if not done:
                predicted = self.model.predict_value(ob)
            # get states, actions, discounted value for last steps 
            states, actions, values = self.env.get_samples(step, predicted)
            # calc loacl and apply remote gradient
            self.model.remote_gradient(states, actions, values)
        return total_steps, total_rewards
    
    
    @staticmethod
    #@profile
    def worker_loop(self):
        sleep(self.id * 2)
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
                self.stat['mean_steps'] = mean_steps
                self.stat['mean_rewards'] = mean_rewards
                self.stat['episode'] = episode
                episode += 1
        except Exception as e:
            self.coord.request_stop(e)
        finally:
            print("Player %d terminated. %d episodes played" % (self.id, episode))
