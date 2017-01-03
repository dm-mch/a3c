#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

from common import Network
from summary import Summary

def linear_decrise_op(config, global_step, name):
    return tf.identity(config['max'] - (config['max'] - config['min']) * tf.minimum(tf.cast(global_step, tf.float32)/float(config['period']), 1), name = name)

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[:3])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = np.prod(filter_shape[:2]) * num_filters
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.zeros_initializer,
                            collections=collections)
        return tf.nn.conv2d(x, w, stride_shape, pad) + b

def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b

def categorical_sample(logits):
    return tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])



class A3CBase(Network):
    """
    Base class for critic (value) and actor (policy) networks

    """
    def __init__(self, name, config, with_summary=True, target=False):
        """
        config: sess, action_dim, state_shape, summary_step, lr, global_step
        config: target_model - only for non target. target=False
        """
        super(A3CBase, self).__init__(name, config.sess)
        self.action_dim = config.action_dim
        self.state_shape = config.state_shape
        self.is_target = target
        self.global_step = config.global_step
        self.eb = config.entropy_beta
        self.clip_grad_norm = config.clip_grad_norm
        # add here tensors for summary
        self.with_summary = with_summary
        self.for_summary_scalar = []
        self.for_summary_hist = []
        self.lr = config.lr['max']#linear_decrise_op(config.lr, self.global_step, 'learning_rate')

        with tf.variable_scope(name):
            self.create_network()

            if not self.is_target:
                self._build_loss()
                self._build_gradient(config.target_model)
                self._build_sync_ops(config.target_model)
                self._build_summary(config.summary_step)

            #else:
                # target should have shared opimizer
                #lr = linear_decrise_op(self.lr, self.global_step, 'learning_rate')
            #    self.optimizer = tf.train.RMSPropOptimizer(self.lr['max'], decay=config.rmsprop_decay, epsilon=config.rmsprop_epsilon)

    def _build_summary(self, summary_step):
        if self.with_summary is not True:
            return

        self.summary = Summary(self.sess, summary_step)
        for s in self.for_summary_scalar:
            self.summary.scalar(s)
        for h in self.for_summary_hist:
            self.summary.hist(h)             
        self.summary.merge()


    def _build_loss(self):
        self.rewards = tf.placeholder(tf.float32, [None])
        self.actions = tf.placeholder(tf.uint8, [None])         
        self.adv = tf.placeholder(tf.float32, [None], name="adv")
   
        a_one_hot = tf.one_hot(self.actions, self.action_dim)

        log_prob = tf.log(self.pf + 1e-6)
        log_pi_a_given_s = tf.reduce_sum(log_prob * a_one_hot, 1)
        policy_loss = -tf.reduce_sum(log_pi_a_given_s * self.adv)
        value_loss = tf.nn.l2_loss(self.vf-self.rewards) # tf.maximum(self.entropy_beta,1)

        entropy_beta = linear_decrise_op(self.eb, self.global_step, 'entropy_beta')
        xentropy_loss = tf.reduce_sum(self.pf * log_prob)

        self.total_loss = policy_loss + 0.5 * value_loss + entropy_beta * xentropy_loss
        batch_size = tf.cast(tf.shape(self.rewards)[0], tf.float32)
        #self.total_loss = tf.truediv(self.total_loss,batch_size,name='total_loss')

        self.for_summary_scalar += [
            tf.reduce_mean(self.adv, name='adv'),
            tf.reduce_mean(self.vf, name='value_mean'),
            tf.reduce_mean(log_pi_a_given_s, name='log_p_mean'),
            tf.reduce_mean(self.rewards, name="true_value_mean"),
            tf.identity(policy_loss/batch_size, name="policy_loss"),
            tf.identity(value_loss/batch_size, name="value_loss"),
            tf.identity((entropy_beta * xentropy_loss)/batch_size, name = 'entropy_loss'),
            entropy_beta,
            # self.lr,
            tf.identity(self.total_loss, name = 'total_loss')
            ]
        self.for_summary_hist += [tf.argmax(self.pf, axis=1, name='action_predicted')]

    def _build_gradient(self, target):
        """
        Local gradient for remote vars

        """
        local_grad = tf.gradients(self.total_loss, self.get_trainable_weights())
        self.for_summary_scalar += [tf.global_norm(local_grad, name='grad_norm'),
                                    tf.global_norm(self.get_trainable_weights(), name='vars_norm')]
        # clip grad by norm
        local_grad, _ = tf.clip_by_global_norm(local_grad, self.clip_grad_norm)
        
        # mix with remote vars
        remote_vars = target.get_trainable_weights()
        assert len(local_grad) == len(remote_vars)
        vars_and_grads = list(zip(local_grad, remote_vars))

        # each worker has a different set of adam optimizer parameters
        optimizer = tf.train.AdamOptimizer(self.lr)

        # apply
        apply_grad = optimizer.apply_gradients(vars_and_grads)
        inc_step = self.global_step.assign_add(tf.shape(self.x)[0])
        self.train_op = tf.group(apply_grad, inc_step)

    def train(self, states, actions, values,  advs, lstm_states):
        if self.is_target:
            raise Exception("remote_gradient only for local networks")

        ops = [self.train_op]
        if self.with_summary:
            # Summary op if needed
            ops += [self.global_step, self.summary.get_op()]

        r = self.sess.run(ops, feed_dict={
            self.x: states,
            self.actions: actions,
            self.rewards: values,
            self.adv: advs,
            self.state_in[0]: lstm_states[0],
            self.state_in[1]: lstm_states[1]
        })

        if self.with_summary and r[-1]:
            self.summary.write(r[-1], global_step = r[-2])

    def get_global_step(self):
        return self.sess.run(self.global_step)

    def _build_sync_ops(self, target):
        self.sync_ops = tf.group(*[v1.assign(v2) for v1, v2 in zip(self.get_trainable_weights(), target.get_trainable_weights())])

    def sync_from_target(self):
        self.sess.run(self.sync_ops)

    def create_network(self):
        """
        should implement
        self.x
        self.vf - value function out
        self.pf - policy function out
        """
        raise NotImplementedError("Method create_networks should be implemented in subclass")

    def get_trainable_weights(self):
        raise NotImplementedError("Method get_trainable_weights should be implemented in subclass")

class A3CLstmNet(A3CBase):
    """
    Simple convolution nets for a3c

    """
    def create_network(self):
        """
        Create Policy and Value network

        """        
        self.x = x = tf.placeholder(tf.float32, [None] + list(self.state_shape))

        for i in range(4):
            x = tf.nn.elu(conv2d(x, 32, "l{}".format(i + 1), [3, 3], [2, 2]))
        # introduce a "fake" batch dimension of 1 after flatten so that we can do LSTM over time dim
        x = tf.expand_dims(flatten(x), [0])

        size = 256
        lstm = tf.nn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
        self.state_size = lstm.state_size
        step_size = tf.shape(self.x)[:1]

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
        self.state_in = [c_in, h_in]

        state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=step_size,
            time_major=False)
        lstm_c, lstm_h = lstm_state
        x = tf.reshape(lstm_outputs, [-1, size])
        self.logits = linear(x, self.action_dim, "action", normalized_columns_initializer(0.01))
        self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
        self.pf = tf.nn.softmax(self.logits)
        self.pa = categorical_sample(self.logits)
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        return self.state_init

    def predict_value(self, state, c, h):
        return self.sess.run(self.vf, {self.x: [state], self.state_in[0]: c, self.state_in[1]: h})[0]

    def predict(self, state, c, h):
        return self.sess.run([self.pa, self.vf] + self.state_out, {self.x: [state], self.state_in[0]: c, self.state_in[1]: h})

    def get_trainable_weights(self):
        return self.var_list


def test():
    from time import time
    from common import Config
    import os 
    os.environ["CUDA_VISIBLE_DEVICES"]=''

    sess = tf.Session()
    global_step = tf.Variable(0, name='global_step', trainable=False)

    config = Config({
        'sess':sess, 
        'action_dim': 3,
        'state_shape': (72,72,4),
        'summary_step': 10,
        'entropy_beta': {'max': 0.1, 'min': 0.1, 'period': 1}, # linear decrise from max to min
        'lr': {'max': 7e-4, 'min':0, 'period': 10 * 10**7},
        'clip_grad_norm': 40.,
        'global_step': global_step,
        'rmsprop_decay': 0.99,
        'rmsprop_epsilon': 0.1,
        })

    target_model = A3CLstmNet("target", config, target = True)
    config['target_model'] = target_model
    model = A3CLstmNet("test", config)

    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    lstm_state = model.get_initial_features()

    pr = model.predict(np.random.randn(72, 72, 4), lstm_state[0], lstm_state[1])
    print('predict length', len(pr), [print(p.shape) for p in pr])
    print("predict", model.predict(np.random.randn(72, 72, 4), lstm_state[0], lstm_state[1]))
    print('value', model.predict_value(np.random.randn(72, 72, 4), lstm_state[0], lstm_state[1]))

    batch_size = 64
    states = np.random.randn(batch_size, 72, 72, 4)
    actions = np.random.randint(0, 3, batch_size)
    values = np.random.randn(batch_size)
    advs = np.random.randn(batch_size)

    target_norm = tf.global_norm(target_model.get_trainable_weights()) 
    local_norm = tf.global_norm(model.get_trainable_weights()) 
    print("before grad target", sess.run(target_norm))
    print("before grad local", sess.run(local_norm))
    print("Calc remote grad")
    saver = target_model.get_saver()
    for j in range(10):
        for i in range(10):
            model.train(states, actions, values,  advs, lstm_state)
        s = time()
        target_model.save(saver)
        print("Save time", time()-s)

    print("after grad target", sess.run(target_norm))
    print("after grad local", sess.run(local_norm))
    model.sync_from_target()
    print("after sync target", sess.run(target_norm))
    print("after sync local", sess.run(local_norm))

    print("Train merged",len(model.get_trainable_weights()))

    # Check saver
    #reset all and load 
    tf.reset_default_graph()
    sess = tf.Session()
    config['sess'] = sess
    target_model = A3CLstmNet("target", config, target = True)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    saver = target_model.get_saver()
    target_model.load(saver)
    target_norm = tf.global_norm(target_model.get_trainable_weights()) 
    print("after reload target", sess.run(target_norm))


if __name__ == '__main__':
    test()