from utils import *
import numpy as np

import tensorflow as tf
import prettytensor as pt
from parameters import pms

seed = 1
np.random.seed(seed)
tf.set_random_seed(seed)

class InnerLSTMCell(tf.nn.rnn_cell.BasicLSTMCell):
    def __init__(self , num_units , forget_bias=1.0 , input_size=None):
        tf.nn.rnn_cell.BasicLSTMCell.__init__(self , num_units , forget_bias=forget_bias , input_size=input_size)
        self.matrix , self.bias = None , None


    def __call__(self , inputs , state , scope=None):
        """
            Long short-term memory cell (LSTM).
            implement from BasicLSTMCell.__call__
        """
        with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
            # Parameters of gates are concatenated into one multiply for efficiency.
            c , h = tf.split(1 , 2 , state)
            concat = self.linear([inputs , h] , 4 * self._num_units , True)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i , j , f , o = tf.split(1 , 4 , concat)

            new_c = c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * tf.tanh(j)
            new_h = tf.tanh(new_c) * tf.sigmoid(o)

            return new_h , tf.concat(1 , [new_c , new_h])


    def linear(self , args , output_size , bias , bias_start=0.0 , scope=None):
        """
            Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
            implement from function of tensorflow.python.ops.rnn_cell.linear()
        """
        if args is None or (isinstance(args , (list , tuple)) and not args):
            raise ValueError("`args` must be specified")
        if not isinstance(args , (list , tuple)):
            args = [args]

            # Calculate the total size of arguments on dimension 1.
        total_arg_size = 0
        shapes = [a.get_shape().as_list() for a in args]
        for shape in shapes:
            if len(shape) != 2:
                raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
            if not shape[1]:
                raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
            else:
                total_arg_size += shape[1]

        # Now the computation.
        with tf.variable_scope(scope or "Linear"):
            matrix = tf.get_variable("Matrix" , [total_arg_size , output_size])
            if len(args) == 1:
                res = tf.matmul(args[0] , matrix)
            else:
                res = tf.matmul(tf.concat(1 , args) , matrix)
            if not bias:
                return res
            bias_term = tf.get_variable(
                "Bias" , [output_size] ,
                initializer=tf.constant_initializer(bias_start))
            self.matrix = matrix
            self.bias = bias_term
        return res + bias_term

class NetworkContinousLSTM(object):
    def __init__(self, scope):
        with tf.variable_scope("%s_shared" % scope):
            self.obs = obs = tf.placeholder(
                dtype, shape=[None, pms.obs_shape], name="%s_obs"%scope)
            self.action_n = tf.placeholder(dtype, shape=[None, pms.action_shape], name="%s_action"%scope)
            self.advant = tf.placeholder(dtype, shape=[None], name="%s_advant"%scope)
            self.old_dist_means_n = tf.placeholder(dtype, shape=[None, pms.action_shape],
                                                   name="%s_oldaction_dist_means"%scope)
            self.old_dist_logstds_n = tf.placeholder(dtype, shape=[None, pms.action_shape],
                                                     name="%s_oldaction_dist_logstds"%scope)
            # self.obs_reshape = tf.reshape(self.obs, [None, 1, pms.obs_shape])
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(3, forget_bias=1.0, state_is_tuple=True)
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob=0.5)
            rnn = tf.nn.rnn_cell.MultiRNNCell([lstm_cell], state_is_tuple=True)
            # rnn = tf.nn.rnn_cell.BasicRNNCell(3)
            self.initial_state = state = rnn.zero_state(tf.shape(self.obs)[0], tf.float32)
            # output , state = tf.nn.dynamic_rnn(rnn, self.obs)
            output, state = rnn(self.obs, state)
            self.action_dist_means_n = (pt.wrap(output).
                                        # fully_connected(64, activation_fn=tf.nn.relu, init=tf.random_normal_initializer(-0.05, 0.05),
                                        #                 name="%s_fc1"%scope).
                                        # fully_connected(64, activation_fn=tf.nn.relu, init=tf.random_normal_initializer(-0.05, 0.05),
                                        #                  name="%s_fc2"%scope).
                                        fully_connected(pms.action_shape, init=tf.random_normal_initializer(-0.05, 0.05),
                                                        name="%s_fc3"%scope))
            self.N = tf.shape(obs)[0]
            Nf = tf.cast(self.N, dtype)
            self.action_dist_logstd_param = tf.Variable((.01*np.random.randn(1, pms.action_shape)).astype(np.float32), trainable=False, name="%spolicy_logstd"%scope)
            self.action_dist_logstds_n = tf.tile(self.action_dist_logstd_param,
                                              tf.pack((tf.shape(self.action_dist_means_n)[0], 1)))
            self.var_list = [v for v in tf.trainable_variables()if v.name.startswith(scope)]

    def get_action_dist_means_n(self, session, obs):
        return session.run(self.action_dist_means_n,
                         {self.obs: obs})

