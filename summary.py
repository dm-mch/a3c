"""
Summary wrapper for tensorboard
Author: Dmitriy Movchan
 
"""

import tensorflow as tf
import os
import shutil


class Summary:

    def __init__(self, sess, perstep ,dir='./logs', force=True):
        self.sess = sess
        if force:
            #clear log dir
            if os.path.isdir(dir):
                print("Clear tensorboard log dir:", dir)
                shutil.rmtree(dir)

        self.perstep = perstep
        self.writer = tf.summary.FileWriter(dir, sess.graph)
        self.i = -1
        self.merged = None
        self.names = []
        self.no_op = tf.no_op(name='nosummary') # Precreate for prevent memory leak

    def merge(self):
        self.merged = tf.summary.merge_all()
        return self.merged

    def get_op(self):
        self.i+=1
        # Merge all the summaries and write them out to log
        if self.merged is None:
            self.merged = self.merge()
        if self.i % self.perstep == 0:
            return self.merged
        else:
            # no summary this step
            return self.no_op

    def scalar(self, *vars):
        for op in vars:
            if op.name not in self.names:
                tf.summary.scalar(op.name, op)
                self.names.append(op.name)

    def hist(self, *tensors):
        for op in tensors:
            if op.name not in self.names:
                tf.summary.histogram(op.name, op)
                self.names.append(op.name)

    def model(self, model):
        for layer in model.layers:
            for weight in layer.weights:
                if weight.name not in self.names:
                    tf.summary.histogram(weight.name, weight)
                    self.names.append(weight.name)

            if hasattr(layer, 'output'):
                name = '{}_out'.format(layer.name)
                if name not in self.names:
                    tf.summary.histogram(name,layer.output)
                    self.names.append(name)

    def write(self, summary, global_step = None):
        if global_step is None:
            global_step = self.i
        self.writer.add_summary(summary, global_step)

    def custom_scalar_summary(self, global_step = None, **argw):
        value = []
        for k,v in argw.items():
            value.append(tf.Summary.Value(tag=str(k), simple_value=v))
        summary = tf.Summary(value=value)
        #if i is None:
        #    i = self.i
        self.writer.add_summary(summary, global_step)