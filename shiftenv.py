#!/usr/bin/env python3

import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray

class Preprocessor:
    """ Fake simple prepocessor """
    def process(self, frame):
        """
        Preprocess each frame of observed states

        """
        return frame

class AtariPreprocessor(Preprocessor):
    """
    Prepocess raw gym atari frame
    """
    def __init__(self, config):
        """
        config: crop, reize
        """
        self.crop = config.crop# crop level
        #self.resize = config.resize
        self.downsample = config.downsample

    def process(self, frame):
        """
        Preprocess obzervation: grayscale, crop, downsample
        """
        crp = self.crop
        dwn = self.downsample
        return np.mean(frame, axis=-1)[crp[0,0]:crp[0,1], crp[1,0]: crp[1,1]][::dwn, ::dwn].astype(np.float32)/128
        #return resize(rgb2gray(frame)[crp[0,0]:crp[0,1], crp[1,0]: crp[1,1]], self.resize)

class ShiftEnv(object):
    """
    Environment wrapper for RL env like gym
    Helps get timeline of played episodes

    """

    def __init__(self, env, config):
        """
        config: preprocessor, skip, tdim, lmbd

        """
        self.env = env
        self.pre = config.preprocessor(config)
        self.skip = config.skip
        self.tdim = config.tdim
        self.lmbd = config.lmbd

        self.gym_actions = range(env.action_space.n)
        if (env.spec.id == "Pong-v0" or env.spec.id == "Breakout-v0"):
            print("Doing workaround for pong or breakout")
            # Gym returns 6 possible actions for breakout and pong.
            # Only three are used, the rest are no-ops. This just lets us
            # pick from a simplified "LEFT", "RIGHT", "NOOP" action space.
            self.gym_actions = [1,2,3]

        self.init()
    
    def init(self):
        self.s_t = None
        self.terminated = False
        self._step = 0
        self.states = []
        self.actions = []
        self.rewards = []


    def reset(self):
        self.init()
        self.s_t = self.pre.process(self.env.reset())
        self.shape = self.s_t.shape + (self.tdim,)
        return self._get_shift(self.s_t)

    def render(self):
        self.env.render()

    def _get_shift(self, current, offset=0):
        """
        Return sample with T deminision. Shape (*current.shape, tdim)

        """
        first = self.states[0] if len(self.states) > 0 else current
        sample =  np.stack([first] * self.tdim, axis=len(current.shape))
        # TODO: make for any deminision state
        sample[:,:,self.tdim-1] = current
        if self._step == 0: 
            return sample
        # revert, skip, cut
        indxs = np.arange(len(self.states) - offset)[::-1][::self.skip][:self.tdim] 
        indxs = indxs[:-1] - (self.tdim - 1)
        if len(indxs) > 0 and len(indxs) < self.tdim-1:
            indxs = indxs.tolist() + [indxs[-1]] * (self.tdim - 1 - len(indxs))

        for f, indx in enumerate(indxs):
            # TODO: make for any deminision state
            sample[:,:,len(indxs) - 1 - f] = self.states[indx]
        return sample

    def step(self, action):
        s_t1, r, self.terminated, info = self.env.step(self.gym_actions[action])
        s_t1 = self.pre.process(s_t1)
        self.states.append(self.s_t)
        self.actions.append(action)
        self.rewards.append(r)
        self._step += 1
        self.s_t = s_t1
        return self._get_shift(s_t1), r, self.terminated, info


    def get_samples(self, n, predicted=0):
        """
        Return tuple (time shifted states, actions, values)
        Calculated for last n or less frames with skip

        """
        # last n, invert, skip, invert
        buffer = self.states[-n:][::-1][::self.skip][::-1]
        # first sample with back time offset
        samples =  [self._get_shift(buffer[0], offset=min(n,len(self.states)))]
        for i in range(1,len(buffer)):
            #TODO: care about different state shape 
            old = np.array(samples[-1], copy = True)[:,:,1:] # All except 1 frame in last sample
            samples.append(np.concatenate((old,buffer[i].reshape(buffer[i].shape[0], buffer[i].shape[1], 1)), axis = 2))
        values = self.get_values(n, predicted)
        actions = self.actions[-n:][::self.skip]
        assert len(samples) == len(values)
        assert len(actions) == len(values)
        return samples, actions, values

    def get_values(self, n, predicted):
        """
        Get value functions by rewards with skip and lambda discounter

        """
        # last n, reverted, clip
        rewards = np.clip(self.rewards[-n:][::-1], -1, 1)
        #print("Rewards: ", rewards)
        value = []
        r = 0 if self.terminated else predicted
        for i in range(len(rewards)):
            r = rewards[i] + self.lmbd * r
            value.append(r)
        # revert back to right way
        value = value[::-1]
        #print("V:",value)
        # add zeros to start for length division on SKIP
        if len(value)%self.skip != 0:
            value = [0] * (self.skip - len(value)%self.skip) + value
        # calc max for "skipped" value
        value = np.max(np.reshape(np.array(value), (int(len(value)/self.skip), self.skip)), axis = 1)
        return value


def test():
    from common import Config
    import gym
    from matplotlib import pyplot as plt

    def show_sample(sample, title = ""):
        t = sample.shape[-1]
        img = np.hstack([sample[:,:,i] for i in range(t)])
        plt.imshow(img)
        plt.title(title)
        plt.show()

    config = Config({
            'preprocessor': AtariPreprocessor,
            'skip': 1,
            'crop': np.array([[48, 192], [8, 152]]),
            'downsample': 2,
            'tdim': 4,
            'lmbd': 0.99
        })

    genv = gym.make('Breakout-v0')
    env = ShiftEnv(genv, config)

    def show_n_samples(n):
        samples, actions, values = env.get_samples(n)
        print(len(samples))
        for i,s in enumerate(samples):
           show_sample(s, "V: " + str(values[i]) + " A: " + str(actions[i]))

    o = env.reset()
    done = False
    step = 0
    while not done:
        o, r, done, info = env.step(np.random.choice(range(3)))
        #if step % 4 == 0:
        #    show_sample(o)
        step+=1
        if step % 100 == 0:
            show_n_samples(100)
    print("Steps:", step)
    show_n_samples(step % 100)


if __name__ == '__main__':
    test()