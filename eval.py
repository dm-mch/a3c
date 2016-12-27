#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import gym
from time import time, sleep

from player import Player
from shiftenv import ShiftEnv, AtariPreprocessor
from ca_network import A3CFFNetwork
from common import Config

# Switch off GPU usage for CUDA, only CPU
import os
#os.environ["CUDA_VISIBLE_DEVICES"]=''


GAME_NAME = 'Breakout-v0'
MODEL_NAME = 'default_model'
# ATARI_RAW_SHAPE = (210, 160, 3)
BREAKOUT_CROP = np.array([[48, 192], [8, 152]])
NO_CROP = np.array([[0, 210], [0, 160]])

TDIM = 4
DOWNSAMPLE = 2
#RESIZE = (84, 84)
SKIP = 1
LAMBDA = 0.99
LR = 7e-4
SUMMARY_STEP = 100
LIMIT_STEP = 1000000
MAX_STEP = 5 * SKIP
ENTROPY_BETA = {'max': 0.01, 'min': 0.001, 'period': 10 * 10**6} # linear decrise from max to min
CLIP_GRAD_NORM = 40.

def get_shape(crop, downsample):
    return (int(round((crop[0,1] - crop[0,0])/downsample)), int(round((crop[1,1] - crop[1,0])/downsample))) 

# workarounds for breakout
def get_crop(game):
    return BREAKOUT_CROP if game == 'Breakout-v0' else NO_CROP

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
        'crop': get_crop(game),
        'state_shape': get_shape(get_crop(game), DOWNSAMPLE) + (TDIM,),
        'summary_step': SUMMARY_STEP,
        'lr': LR,
        'global_step': global_step,
        'preprocessor': AtariPreprocessor,
        'skip': SKIP,
        'downsample': DOWNSAMPLE,
        'tdim': TDIM,
        'lmbd': LAMBDA,
        'coord': coord,
        'limit_step': LIMIT_STEP,
        'max_step': MAX_STEP,
        'entropy_beta': ENTROPY_BETA,
        'clip_grad_norm': CLIP_GRAD_NORM,
        'render': render
        })

    print(config)

    target_model = A3CFFNetwork("target", config, target = True, with_summary=False)
    config['target_model'] = target_model
    if load_weights:
        try:
            target_model.load_weights()
        except:
            print("Не удалось загурзить веса, инициалзируем начальными")

    stat = [dict() for i in range(n_threads)]
    config['stat'] = stat
    envs = [ShiftEnv(gym.make(game), config) for i in range(n_threads)]
    models = [A3CFFNetwork("Player_" + str(i), config, with_summary=(i==0)) for i in range(n_threads)]
    players = [Player(i, envs[i], models[i], config)  for i in range(n_threads)]

    # init all tf vars
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # start all players
    for p in players: p.start()

    last_check = time()
    last_save = time()
    gs = 0
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
                target_model.save()
    except Exception as e:
        print("Report exceptions to the coordinator.", e)
        coord.request_stop(e)
    finally:
        print("Terminate as usual. Waiting for threads")
        coord.request_stop()
        coord.join(players)

if __name__ == '__main__':
    train(8, GAME_NAME, 10 * 10**6, render=False, load_weights = False)

