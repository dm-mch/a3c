#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import gym
from time import time, sleep

from player import Player
from ca_network import A3CLstmNet
from common import Config

# Switch off GPU usage for CUDA, only CPU
import os
#os.environ["CUDA_VISIBLE_DEVICES"]=''


GAME_NAME = 'Pong-v0'


def get_action_dim(game):
    return 3 if game == 'Breakout-v0' or game == "Pong-v0" else 6

def prepare_stat(stat):
    for_sum = ['episode']
    total = {}
    for v in stat[0].keys():
        total[v] = [stat[i].get(v, 0) for i in range(len(stat))]
        if v in for_sum:
            total[v] = float(np.sum(total[v]))
        else:
            total[v] = float(np.mean(total[v]))
    return total


def train(n_threads, game, steps, render=True, load_weights = True):

    sess = tf.Session()
    
    # for count train steps
    global_step = tf.Variable(0, name='global_step', trainable=False)
    coord = tf.train.Coordinator()

    config = Config({
        'sess':sess, 
        'action_dim': get_action_dim(game),
        'state_shape': (42, 42, 1),
        'summary_step': 10,
        'lr': {'max': 1e-4, 'min':0, 'period': 10**7},
        'global_step': global_step,
        'gamma': 0.99,
        'coord': coord,
        'limit_step': 10**7,
        'max_step': 20,
        'entropy_beta': {'max': 0.01, 'min': 0.01, 'period': 1}, # linear decrise from max to min
        'clip_grad_norm': 40.,
        #'rmsprop_decay': 0.99,
        #'rmsprop_epsilon': 0.1,
        'actions': [1,2,3] if get_action_dim(game) == 3 else list(range(get_action_dim(game))),
        'render': render
        })

    print(config)

    target_model = A3CLstmNet("target", config, target = True, with_summary=False)
    config['target_model'] = target_model
    if load_weights:
        try:
            target_model.load_weights()
        except:
            print("Не удалось загурзить веса, инициалзируем начальными")

    stat = [dict() for i in range(n_threads)]
    config['stat'] = stat
    envs = [gym.make(game) for i in range(n_threads)]
    models = [A3CLstmNet("Player_" + str(i), config, with_summary=(i==0)) for i in range(n_threads)]
    players = [Player(i, envs[i], models[i], config, render = render and (i==0))  for i in range(n_threads)]

    # init all tf vars
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # start all players
    for p in players: p.start()

    last_check = time()
    last_save = time()
    gs = 0
    saver = target_model.get_saver()
    try:
        while True:
            sleep(0.1)
            if coord.should_stop():
                print("Coord should stop!")
                break

            if time() - last_check > 3: # sec.
                last_gs = gs
                gs = sess.run(global_step)
                if gs >= steps:
                    print("All steps completed!")
                    break
                cur_stat = prepare_stat(stat)
                print("Step", gs, "Time per step %.4f" % ((time() - last_check)/(gs - last_gs)),
                      ", ".join([k + ': ' + '%.2f'%(float(v)) for k,v in cur_stat.items()]))
                models[0].summary.custom_scalar_summary(global_step = gs, **cur_stat)
                last_check = time()
            if time() - last_save > 60:
                last_save = time()
                target_model.save(saver)
    except Exception as e:
        print("Report exceptions to the coordinator.", e)
        coord.request_stop(e)
    finally:
        print("Terminate as usual. Waiting for threads")
        coord.request_stop()
        coord.join(players)

if __name__ == '__main__':
    train(8, GAME_NAME, 10**7, render=True, load_weights = False)

