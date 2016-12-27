import os
import numpy as np


def make_action(policy_value):
    return np.random.choice(np.arange(len(policy_value)), p = policy_value)

class Config(dict):
    def __getattr__(self, attr):
        if len(attr) > 2 and attr[:2] == 's_':
            # take safe
            return self.get(attr[2:], None)
        return self[attr]

class Network:
    """
    Base network collection class
    """
    def __init__(self, name):
        self.net = {} # all keras model should be hear
        self.name = name
        self.folder = "weights"


    def get_path(self, folder=True, ext=".model", ind=None):
        root_path = os.path.join(os.path.curdir, self.folder)
        if not os.path.exists(root_path):
            os.mkdir(root_path)
        model_path = os.path.join(root_path, self.name)
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        if folder:
            return model_path

        if ind is None:
            ind = ''
        else:
            ind = '_' + str(ind)
        return os.path.join(model_path, ind + ext)


    def load_weights(self, ind=None, by_name=False):
        for name, model in self.net.items():
            try:    
                path = self.get_path(folder=False, ext=name + '.hd5', ind=ind)
                model.load_weights(path, by_name = by_name)
                print("Loaded weights for model: ", path)
            except FileNotFoundError as e:
                print("Weights not found: ", path)
                continue

    def save_weights(self, ind=None):
        for name, model in self.net.items():
            path = self.get_path(folder=False, ext=name + '.hd5', ind=ind)
            model.save_weights(path)

    def load(self, ind=None):
        for name, model in self.net.items():
            try:    
                path = self.get_path(folder=False, ext=name + '.model.hd5', ind=ind)
                self.models[name] = load_model(path)
                print("Loaded model: ", path)
            except FileNotFoundError as e:
                print("File for model not found: ", path)
                continue

    def save(self, ind=None):
        for name, model in self.net.items():
            path = self.get_path(folder=False, ext=name + '.model.hd5', ind=ind)
            model.save(path)
