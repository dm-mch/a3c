import os
import numpy as np
import tensorflow as tf


def make_action(policy_value):
    return np.random.choice(np.arange(len(policy_value)), p = policy_value)

class Config(dict):
    def __getattr__(self, attr):
        if len(attr) > 2 and attr[:2] == 's_':
            # take safe
            return self.get(attr[2:], None)
        return self[attr]

# Disables write_meta_graph argument, which freezes entire process and is mostly useless.
class FastSaver(tf.train.Saver):
    def save(self, sess, save_path, global_step=None, latest_filename=None,
             meta_graph_suffix="meta", write_meta_graph=True):
        super(FastSaver, self).save(sess, save_path, global_step, latest_filename,
                                    meta_graph_suffix, False)


class Network:
    """
    Base network collection class
    """
    def __init__(self, name, sess):
        self.name = name
        self.sess = sess
        self.folder = "weights"

    def get_saver(self, additional=[]):
        variables_to_save = [v for v in tf.global_variables() if v.name.startswith(self.name)]
        print("Saver: Vars for save:", len(variables_to_save), [v.name for v in variables_to_save])
        variables_to_save += additional
        return FastSaver(variables_to_save)

    def get_path(self, folder=True, ext=".model"):
        root_path = os.path.join(os.path.curdir, self.folder)
        if not os.path.exists(root_path):
            os.mkdir(root_path)
        model_path = os.path.join(root_path, self.name)
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        if folder:
            return model_path

        return os.path.join(model_path, ext)

    def load(self, saver):
        try:    
            path = self.get_path(folder=True)
            path = tf.train.latest_checkpoint(path)
            saver.restore(self.sess, path)
            print("Loaded model: ", path)
            return True
        except FileNotFoundError as e:
            print("File for model not found: ", path)
            return False

    def save(self, saver):
        path = self.get_path(folder=False, ext=self.name + '.model')
        saver.save(self.sess, path)
