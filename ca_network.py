#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

from keras import backend as K
from keras.layers import Dense, Input, Convolution2D, Permute, Flatten, MaxPooling2D, BatchNormalization, Activation
from keras.models import Model

from common import Network
from summary import Summary

class A3CBase(Network):
    """
    Base class for critic (value) and actor (policy) networks

    """
    def __init__(self, name, config, with_summary=True, target=False):
        """
        config: sess, action_dim, state_shape, summary_step, lr, global_step
        config: target_model - only for non target. target=False
        """
        super(A3CBase, self).__init__(name)
        self.sess = config.sess
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

        self.states = tf.placeholder("float", [None] + list(self.state_shape))
        self.net['policy'], self.net['value'] = self.create_networks()
        self.v_out = self.net['value'](self.states)
        self.p_out = self.net['policy'](self.states)

        K.set_session(self.sess)

        if not self.is_target:
            self._build_loss()
            self._build_gradient(config.target_model)
            self._build_sync_ops(config.target_model)
            self._build_summary(config.summary_step)
        else:
            # target should have shared opimizer
            self.optimizer = tf.train.RMSPropOptimizer(config.lr, decay=0.99, epsilon=0.1)


    def _build_summary(self, summary_step):
        if self.with_summary is not True:
            return

        self.summary = Summary(self.sess, summary_step)
        for s in self.for_summary_scalar:
            self.summary.scalar(s)
        for h in self.for_summary_hist:
            self.summary.hist(h)             
        for net in self.net.values():
            self.summary.model(net)
        self.summary.merge()


    def _build_loss(self):
        self.rewards = tf.placeholder("float", [None])
        self.actions = tf.placeholder("uint8", [None])            


        a_one_hot = tf.one_hot(self.actions, self.action_dim)
        value = tf.squeeze(self.v_out, [1], name='pred_value') # remove 1 dim

        log_prob = tf.log(self.p_out + 1e-6)
        advantage = tf.sub(self.rewards, value, name='advantage')
        entropy_beta = tf.identity(self.eb['max'] - (self.eb['max'] - self.eb['min']) * tf.minimum(tf.cast(self.global_step, tf.float32)/float(self.eb['period']), 1), name = 'entropy_beta')
        xentropy_loss = -tf.reduce_sum(self.p_out * log_prob, 1)
        log_pi_a_given_s = tf.reduce_sum(log_prob * a_one_hot, 1)
        policy_loss = -tf.reduce_sum(log_pi_a_given_s * advantage  + entropy_beta * xentropy_loss)
        value_loss = 0.5 * tf.nn.l2_loss(advantage) # tf.maximum(self.entropy_beta,1)

        self.total_loss = policy_loss + value_loss# + entropy_beta * xentropy_loss
        #batch_size = tf.cast(tf.shape(self.rewards)[0], tf.float32)
        #self.total_loss = tf.truediv(self.total_loss,batch_size,name='total_loss')

        if self.with_summary:
            self.for_summary_scalar += [
                tf.reduce_mean(advantage, name='adv'),
                tf.reduce_mean(value, name='value_mean'),
                tf.reduce_mean(log_pi_a_given_s, name='log_p_mean'),
                tf.reduce_mean(self.rewards, name="true_value_mean"),
                tf.identity(policy_loss, name="policy_loss"),
                tf.identity(value_loss, name="value_loss"),
                tf.identity(tf.reduce_sum(entropy_beta * xentropy_loss), name = 'entropy_loss'),
                entropy_beta,
                tf.identity(self.total_loss, name = 'total_loss')
                ]
            self.for_summary_hist += [tf.argmax(self.p_out, axis=1, name='action_predicted')]

    def _build_gradient(self, target):
        """
        Local gradient for remote vars

        """
        local_grad = tf.gradients(self.total_loss, self.get_trainable_weights(),
                                    gate_gradients=False,
                                    aggregation_method=None,
                                    colocate_gradients_with_ops=False)
        # clip grad by norm
        for i in range(len(local_grad)):
            local_grad[i] = tf.clip_by_norm(local_grad[i], self.clip_grad_norm)
        
        # mix with remote vars
        remote_vars = target.get_trainable_weights()
        assert len(local_grad) == len(remote_vars)
        vars_and_grads = []
        for i,rv in enumerate(remote_vars):
            vars_and_grads.append((local_grad[i], rv))
        # apply
        self.apply_grad = target.optimizer.apply_gradients(vars_and_grads, global_step = self.global_step)

        if self.with_summary:
            self.for_summary_scalar += [tf.reduce_mean([tf.reduce_mean(g) for g in local_grad], name = 'gradient_mean')]

    def remote_gradient(self, states, actions, values):
        if self.is_target:
            raise Exception("remote_gradient only for local networks")

        ops = [self.apply_grad]
        if self.with_summary:
            # Summary op if needed
            ops += [self.global_step, self.summary.get_op()]

        r = self.sess.run(ops, feed_dict={
            self.states: states,
            self.actions: actions,
            self.rewards: values
        })

        if self.with_summary and r[-1]:
            self.summary.write(r[-1], global_step = r[-2])

    def get_global_step(self):
        return self.sess.run(self.global_step)


    def _build_sync_ops(self, target):
        self.sync_ops = []
        for name, local in self.net.items():
            local_weights = local.weights
            target_weights = target.net[name].weights
            local_weights.sort(key=lambda x: x.name)
            target_weights.sort(key=lambda x: x.name)
            assert len(local_weights) == len(target_weights)
            for i in range(len(local_weights)):
                self.sync_ops.append(tf.assign(local_weights[i], target_weights[i]))

    def sync_from_target(self):
        self.sess.run(self.sync_ops)


    def predict_policy(self, state):
        return self.sess.run(self.p_out, feed_dict={
            self.states: np.array([state])
        })[0]

    def predict_value(self, state):
        return self.sess.run(self.v_out, feed_dict={
            self.states: np.array([state])
        })[0]

    def create_networks(self):
        """
        Return policy and value keras model
        """
        raise NotImplemented("Method create_networks should be implemented in subclass")

    def get_trainable_weights(self):
        """
        Models can have intersected set of train vars,
        For conpute gradint we should prepare only unique var by its name

        """
        trainable_weights = []
        names = set()
        for model in self.net.values():
            for w in model.trainable_weights:
                if w.name not in names:
                    names.add(w.name)
                    trainable_weights.append(w)
        # sort by name
        trainable_weights.sort(key=lambda x: x.name)
        return trainable_weights


class A3CFFNetwork(A3CBase):
    """
    Simple convolution nets for a3c

    """
    def create_networks(self):
        """
        Create Policy and Value net

        """        
        with tf.name_scope(self.name + "_nets"):
            input = Input(shape=self.state_shape, tensor=self.states) # should by (x, y, t), where t is frames
        
            shared = Convolution2D(16, 8,8, subsample = (4,4), activation='relu',  border_mode='same', name="conv1",)(input)
            shared = Convolution2D(32,4,4, subsample = (2,2), activation='relu', border_mode='same', name="conv2")(shared)
            
            shared = Flatten()(shared)
            shared = Dense(name="fc0", output_dim=256, activation='relu')(shared)

            policy = Dense(name="policy_softmax", output_dim=self.action_dim, activation='softmax')(shared)
            value = Dense(name="value", output_dim=1)(shared)

        policy_network = Model(input=input, output=policy)
        value_network = Model(input=input, output=value)

        return policy_network, value_network


    # def create_networks(self):
    #     """
    #     Create Policy and Value net

    #     """        
    #     with tf.name_scope(self.name + "_nets"):
    #         input = Input(shape=self.state_shape, tensor=self.states) # should by (x, y, t), where t is frames
        
    #         shared = Convolution2D(name="conv1", nb_filter=32, nb_row=5, nb_col=5, border_mode='same')(input)
    #         shared = Activation('relu')(shared)
    #         shared = MaxPooling2D(pool_size=(2, 2))(shared)

    #         shared = Convolution2D(name="conv2", nb_filter=32, nb_row=5, nb_col=5, border_mode='same')(shared)
    #         shared = Activation('relu')(shared)
    #         shared = MaxPooling2D(pool_size=(2, 2))(shared)

    #         shared = Convolution2D(name="conv3", nb_filter=64, nb_row=4, nb_col=4, border_mode='same')(shared)
    #         shared = Activation('relu')(shared)
    #         shared = MaxPooling2D(pool_size=(2, 2))(shared)

    #         shared = Convolution2D(name="conv4", nb_filter=64, nb_row=3, nb_col=3, border_mode='same')(shared)
    #         shared = Activation('relu')(shared)
            
    #         shared = Flatten()(shared)
    #         shared = Dense(name="fc0", output_dim=512, activation='relu')(shared)

    #         policy = Dense(name="policy_linear", output_dim=self.action_dim)(shared)
    #         policy = Activation('softmax', name="policy_softmax")(policy)
    #         value = Dense(name="value", output_dim=1)(shared)

    #     policy_network = Model(input=input, output=policy)
    #     value_network = Model(input=input, output=value)

    #     return policy_network, value_network



def test():
    from common import Config
    sess = tf.Session()
    config = Config({
        'sess':sess, 
        'action_dim': 3,
        'state_shape': (72,72,4),
        'summary_step': 10,
        'lr': 0.001,
        })
    target_model = A3CFFNetwork("target", config, target = True)
    config['target_model'] = target_model
    model = A3CFFNetwork("test", config)


    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print("policy", model.predict_policy(np.random.randn(72, 72, 4)))
    print('value', model.predict_value(np.random.randn(72, 72, 4)))

    batch_size = 64
    states = np.random.randn(batch_size, 72, 72, 4)
    actions = np.random.randint(0, 3, batch_size)
    values = np.random.randn(batch_size)

    target_sum_weights = tf.reduce_sum(target_model.get_trainable_weights()[-1]) 
    local_sum_weights = tf.reduce_sum(model.get_trainable_weights()[-1]) 
    print("before grad target", sess.run(target_sum_weights))
    print("before grad local", sess.run(local_sum_weights))
    print("Calc remote grad")
    model.remote_gradient(states, actions, values, 0.1)
    print("after grad target", sess.run(target_sum_weights))
    print("after grad local", sess.run(local_sum_weights))
    model.sync_from_target()
    print("after sync target", sess.run(target_sum_weights))
    print("after sync local", sess.run(local_sum_weights))

    print("Train w policy",len(model.net['policy'].trainable_weights))
    print("Train w value",len(model.net['value'].trainable_weights))
    print("Train merged",len(model.get_trainable_weights()))


if __name__ == '__main__':
    test()