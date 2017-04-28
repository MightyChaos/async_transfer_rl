# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

# Actor-Critic Network Base Class
# (Policy network and Value network)
class GameACNetwork(object):
  def __init__(self,
               action_size1,
               action_size2,
               thread_index, # -1 for global               
               device="/cpu:0"):
    self._action_size1 = action_size1
    self._action_size2 = action_size2
    self._thread_index = thread_index
    self._device = device    

  def prepare_loss(self, entropy_beta):
    with tf.device(self._device):
      # taken action (input for policy)
      self.a1 = tf.placeholder("float", [None, self._action_size1])
      self.a2 = tf.placeholder("float", [None, self._action_size2])

      # temporary difference (R-V) (input for policy)
      self.td1 = tf.placeholder("float", [None])
      self.td2 = tf.placeholder("float", [None])

      # avoid NaN with clipping when value in pi becomes zero
      log_pi1 = tf.log(tf.clip_by_value(self.pi1, 1e-20, 1.0))
      log_pi2 = tf.log(tf.clip_by_value(self.pi2, 1e-20, 1.0))

      # policy entropy
      entropy1 = -tf.reduce_sum(self.pi1 * log_pi1, reduction_indices=1)
      entropy2 = -tf.reduce_sum(self.pi2 * log_pi2, reduction_indices=1)

      # policy loss (output)  (Adding minus, because the original paper's objective function is for gradient ascent, but we use gradient descent optimizer.)
      policy_loss1 = - tf.reduce_sum( tf.reduce_sum( tf.multiply( log_pi1, self.a1 ), reduction_indices=1 ) * self.td1 + entropy1 * entropy_beta )
      policy_loss2 = - tf.reduce_sum( tf.reduce_sum( tf.multiply( log_pi2, self.a2 ), reduction_indices=1 ) * self.td2 + entropy2 * entropy_beta )

      # R (input for value)
      self.r1 = tf.placeholder("float", [None])
      self.r2 = tf.placeholder("float", [None])

      # value loss (output)
      # (Learning rate for Critic is half of Actor's, so multiply by 0.5)
      value_loss1 = 0.5 * tf.nn.l2_loss(self.r1 - self.v1)
      value_loss2 = 0.5 * tf.nn.l2_loss(self.r2 - self.v2)

      # gradienet of policy and value are summed up
      self.total_loss1 = policy_loss1 + value_loss1
      self.total_loss2 = policy_loss2 + value_loss2

  def run_policy_and_value(self, sess, s_t, game_index):
    raise NotImplementedError()
    
  def run_policy(self, sess, s_t, game_index):
    raise NotImplementedError()

  def run_value(self, sess, s_t, game_index):
    raise NotImplementedError()    

  def get_vars(self):
    raise NotImplementedError()

  def sync_from(self, src_netowrk, name=None):
    src_vars = src_netowrk.get_vars()
    dst_vars = self.get_vars()

    sync_ops = []

    with tf.device(self._device):
      with tf.name_scope(name, "GameACNetwork", []) as name:
        for(src_var, dst_var) in zip(src_vars, dst_vars):
          sync_op = tf.assign(dst_var, src_var)
          sync_ops.append(sync_op)

        return tf.group(*sync_ops, name=name)

  # weight initialization based on muupan's code
  # https://github.com/muupan/async-rl/blob/master/a3c_ale.py
  def _fc_variable(self, weight_shape):
    input_channels  = weight_shape[0]
    output_channels = weight_shape[1]
    d = 1.0 / np.sqrt(input_channels)
    bias_shape = [output_channels]
    weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
    bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d))
    return weight, bias

  def _conv_variable(self, weight_shape):
    w = weight_shape[0]
    h = weight_shape[1]
    input_channels  = weight_shape[2]
    output_channels = weight_shape[3]
    d = 1.0 / np.sqrt(input_channels * w * h)
    bias_shape = [output_channels]
    weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
    bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d))
    return weight, bias

  def _conv2d(self, x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

# Actor-Critic FF Network
class GameACFFNetwork(GameACNetwork):
  def __init__(self,
               action_size1,
               action_size2,
               thread_index, # -1 for global
               device="/cpu:0"):
    GameACNetwork.__init__(self, action_size1, action_size2, thread_index, device)

    scope_name = "net_" + str(self._thread_index)
    with tf.device(self._device), tf.variable_scope(scope_name) as scope:
      self.W_conv1, self.b_conv1 = self._conv_variable([8, 8, 4, 16])  # stride=4
      self.W_conv2, self.b_conv2 = self._conv_variable([4, 4, 16, 32]) # stride=2

      self.W_fc1, self.b_fc1 = self._fc_variable([2592, 256])

      # weight for policy output layer
      self.W_fc2_1, self.b_fc2_1 = self._fc_variable([256, action_size1])
      self.W_fc2_2, self.b_fc2_2 = self._fc_variable([256, action_size2])

      # weight for value output layer
      self.W_fc3_1, self.b_fc3_1 = self._fc_variable([256, 1])
      self.W_fc3_2, self.b_fc3_2 = self._fc_variable([256, 1])

      # state (input)
      self.s = tf.placeholder("float", [None, 84, 84, 4])
    
      h_conv1 = tf.nn.relu(self._conv2d(self.s,  self.W_conv1, 4) + self.b_conv1)
      h_conv2 = tf.nn.relu(self._conv2d(h_conv1, self.W_conv2, 2) + self.b_conv2)

      h_conv2_flat = tf.reshape(h_conv2, [-1, 2592])
      #two streams of fc layer

      h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, self.W_fc1) + self.b_fc1)

      # policy (output)
      self.pi1 = tf.nn.softmax(tf.matmul(h_fc1, self.W_fc2_1) + self.b_fc2_1)
      self.pi2 = tf.nn.softmax(tf.matmul(h_fc1, self.W_fc2_2) + self.b_fc2_2)
      # value (output)
      v_1 = tf.matmul(h_fc1, self.W_fc3_1) + self.b_fc3_1
      v_2 = tf.matmul(h_fc1, self.W_fc3_2) + self.b_fc3_2
      self.v1 = tf.reshape( v_1, [-1] )
      self.v2 = tf.reshape( v_2, [-1] )

  def run_policy_and_value(self, sess, s_t, game_index):
    if game_index == 1:
      pi_out, v_out = sess.run( [self.pi1, self.v1], feed_dict = {self.s : [s_t]} )
    elif game_index == 2:
      pi_out, v_out = sess.run( [self.pi2, self.v2], feed_dict = {self.s : [s_t]} )
    else:
      raise ValueError("game_index should be 1 or 2")
    return (pi_out[0], v_out[0])

  def run_policy(self, sess, s_t, game_index):
    if game_index == 1:
      pi_out = sess.run( self.pi1, feed_dict = {self.s : [s_t]} )
    elif game_index == 2:
      pi_out = sess.run( self.pi2, feed_dict = {self.s : [s_t]} )
    else:
      raise ValueError("game_index should be 1 or 2")
    return pi_out[0]

  def run_value(self, sess, s_t, game_index):
    if game_index == 1:
      v_out = sess.run( self.v1, feed_dict = {self.s : [s_t]} )
    elif game_index == 2:
      v_out = sess.run( self.v2, feed_dict = {self.s : [s_t]} )
    else:
      raise ValueError("game_index should be 1 or 2")
    return v_out[0]

  def get_vars(self):
    return [self.W_conv1, self.b_conv1,
            self.W_conv2, self.b_conv2,
            self.W_fc1, self.b_fc1,
            self.W_fc2_1, self.b_fc2_1,
            self.W_fc2_2, self.b_fc2_2,
            self.W_fc3_1, self.b_fc3_1,
            self.W_fc3_2, self.b_fc3_2]



  def get_vars_1(self):
    return [self.W_conv1, self.b_conv1,
            self.W_conv2, self.b_conv2,
            self.W_fc1, self.b_fc1,
            self.W_fc2_1, self.b_fc2_1,
            self.W_fc3_1, self.b_fc3_1]

  def get_vars_2(self):
    return [self.W_conv1, self.b_conv1,
            self.W_conv2, self.b_conv2,
            self.W_fc1, self.b_fc1,
            self.W_fc2_2, self.b_fc2_2,
            self.W_fc3_2, self.b_fc3_2]










# Actor-Critic LSTM Network
class GameACLSTMNetwork(GameACNetwork):
  def __init__(self,
               action_size1,
               action_size2,
               thread_index, # -1 for global
               device="/cpu:0" ):
    GameACNetwork.__init__(self, action_size1, action_size2, thread_index, device)

    scope_name = "net_" + str(self._thread_index)
    with tf.device(self._device), tf.variable_scope(scope_name) as scope:
      self.W_conv1, self.b_conv1 = self._conv_variable([8, 8, 4, 16])  # stride=4
      self.W_conv2, self.b_conv2 = self._conv_variable([4, 4, 16, 32]) # stride=2
      
      self.W_fc1, self.b_fc1 = self._fc_variable([2592, 256])

      # lstm
      self.lstm_1 = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)
      self.lstm_2 = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)

      # weight for policy output layer
      self.W_fc2_1, self.b_fc2_1 = self._fc_variable([256, action_size1])
      self.W_fc2_2, self.b_fc2_2 = self._fc_variable([256, action_size2])

      # weight for value output layer
      self.W_fc3_1, self.b_fc3_1 = self._fc_variable([256, 1])
      self.W_fc3_2, self.b_fc3_2 = self._fc_variable([256, 1])

      # state (input)
      self.s = tf.placeholder("float", [None, 84, 84, 4])
    
      h_conv1 = tf.nn.relu(self._conv2d(self.s,  self.W_conv1, 4) + self.b_conv1)
      h_conv2 = tf.nn.relu(self._conv2d(h_conv1, self.W_conv2, 2) + self.b_conv2)

      h_conv2_flat = tf.reshape(h_conv2, [-1, 2592])
      h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, self.W_fc1) + self.b_fc1)
      # h_fc1 shape=(5,256)

      h_fc1_reshaped = tf.reshape(h_fc1, [1,-1,256])
      # h_fc_reshaped = (1,5,256)

      # place holder for LSTM unrolling time step size.
      #TODO: do we need two placeholders for lstm states or stepsize?
      self.step_size_1 = tf.placeholder(tf.float32, [1])
      self.step_size_2 = tf.placeholder(tf.float32, [1])

      self.initial_lstm_state0_1 = tf.placeholder(tf.float32, [1, 256])
      self.initial_lstm_state1_1 = tf.placeholder(tf.float32, [1, 256])
      self.initial_lstm_state_1 = tf.contrib.rnn.LSTMStateTuple(self.initial_lstm_state0_1,
                                                              self.initial_lstm_state1_1)

      self.initial_lstm_state0_2 = tf.placeholder(tf.float32, [1, 256])
      self.initial_lstm_state1_2 = tf.placeholder(tf.float32, [1, 256])
      self.initial_lstm_state_2 = tf.contrib.rnn.LSTMStateTuple(self.initial_lstm_state0_2,
                                                              self.initial_lstm_state1_2)


      # Unrolling LSTM up to LOCAL_T_MAX time steps. (= 5time steps.)
      # When episode terminates unrolling time steps becomes less than LOCAL_TIME_STEP.
      # Unrolling step size is applied via self.step_size placeholder.
      # When forward propagating, step_size is 1.
      # (time_major = False, so output shape is [batch_size, max_time, cell.output_size])
    scope_name_lstm1 = "net_" + str(self._thread_index) + 'lstm_1'
    with tf.device(self._device), tf.variable_scope(scope_name_lstm1) as scope:
      lstm_outputs_1, self.lstm_state_1 = tf.nn.dynamic_rnn(self.lstm_1,
                                                        h_fc1_reshaped,
                                                        initial_state = self.initial_lstm_state_1,
                                                        sequence_length = self.step_size_1,
                                                        time_major = False,
                                                        scope = scope)
      scope.reuse_variables()
      self.W_lstm_1 = tf.get_variable("basic_lstm_cell/weights")
      self.b_lstm_1 = tf.get_variable("basic_lstm_cell/biases")


    scope_name_lstm2 = "net_" + str(self._thread_index) + 'lstm_2'
    with tf.device(self._device), tf.variable_scope(scope_name_lstm2) as scope:
      lstm_outputs_2, self.lstm_state_2 = tf.nn.dynamic_rnn(self.lstm_2,
                                                        h_fc1_reshaped,
                                                        initial_state = self.initial_lstm_state_2,
                                                        sequence_length = self.step_size_2,
                                                        time_major = False,
                                                        scope = scope)
      scope.reuse_variables()
      self.W_lstm_2 = tf.get_variable("basic_lstm_cell/weights")
      self.b_lstm_2 = tf.get_variable("basic_lstm_cell/biases")

      # lstm_outputs: (1,5,256) for back prop, (1,1,256) for forward prop.
    scope_name = "net_" + str(self._thread_index)
    with tf.device(self._device), tf.variable_scope(scope_name) as scope:
      lstm_outputs_1 = tf.reshape(lstm_outputs_1, [-1,256])
      lstm_outputs_2 = tf.reshape(lstm_outputs_2, [-1,256])

      # policy (output)
      self.pi1 = tf.nn.softmax(tf.matmul(lstm_outputs_1, self.W_fc2_1) + self.b_fc2_1)
      self.pi2 = tf.nn.softmax(tf.matmul(lstm_outputs_2, self.W_fc2_2) + self.b_fc2_2)

      # value (output)
      v_1_ = tf.matmul(lstm_outputs_1, self.W_fc3_1) + self.b_fc3_1
      v_2_ = tf.matmul(lstm_outputs_2, self.W_fc3_2) + self.b_fc3_2
      self.v1 = tf.reshape( v_1_, [-1] )
      self.v2 = tf.reshape( v_2_, [-1] )

      #scope.reuse_variables()
      #self.W_lstm = tf.get_variable("basic_lstm_cell/weights")
      #self.b_lstm = tf.get_variable("basic_lstm_cell/biases")

      self.reset_state()
      
  def reset_state(self):
    self.lstm_state_out = tf.contrib.rnn.LSTMStateTuple(np.zeros([1, 256]),
                                                        np.zeros([1, 256]))

  def run_policy_and_value(self, sess, s_t, game_index):
    # This run_policy_and_value() is used when forward propagating.
    # so the step size is 1.
    if game_index == 1:
        pi_out, v_out, self.lstm_state_out = sess.run( [self.pi1, self.v1, self.lstm_state_1],
                                                   feed_dict = {self.s : [s_t],
                                                                self.initial_lstm_state0_1 : self.lstm_state_out[0],
                                                                self.initial_lstm_state1_1 : self.lstm_state_out[1],
                                                                self.step_size_1 : [1]} )

    elif game_index == 2:
        pi_out, v_out, self.lstm_state_out = sess.run( [self.pi2, self.v2, self.lstm_state_2],
                                                   feed_dict = {self.s : [s_t],
                                                                self.initial_lstm_state0_2 : self.lstm_state_out[0],
                                                                self.initial_lstm_state1_2 : self.lstm_state_out[1],
                                                                self.step_size_2 : [1]} )
    else:
        raise ValueError("game index out of range")
    # pi_out: (1,3), v_out: (1)
    return (pi_out[0], v_out[0])

  def run_policy(self, sess, s_t, game_index):
    # This run_policy() is used for displaying the result with display tool.    
    if game_index == 1:
        pi_out, self.lstm_state_out = sess.run( [self.pi1, self.lstm_state_1],
                                            feed_dict = {self.s : [s_t],
                                                         self.initial_lstm_state0_1 : self.lstm_state_out[0],
                                                         self.initial_lstm_state1_1 : self.lstm_state_out[1],
                                                         self.step_size_1 : [1]} )
    elif game_index == 2:
        pi_out, self.lstm_state_out = sess.run( [self.pi2, self.lstm_state_2],
                                            feed_dict = {self.s : [s_t],
                                                         self.initial_lstm_state0_2 : self.lstm_state_out[0],
                                                         self.initial_lstm_state1_2 : self.lstm_state_out[1],
                                                         self.step_size_2 : [1]} )
    else:
        raise ValueError("game index out of range")
    return pi_out[0]

  def run_value(self, sess, s_t, game_index):
    # This run_value() is used for calculating V for bootstrapping at the 
    # end of LOCAL_T_MAX time step sequence.
    # When next sequcen starts, V will be calculated again with the same state using updated network weights,
    # so we don't update LSTM state here.
    prev_lstm_state_out = self.lstm_state_out
    if game_index == 1:
        v_out, _ = sess.run( [self.v1, self.lstm_state_1],
                         feed_dict = {self.s : [s_t],
                                      self.initial_lstm_state0_1 : self.lstm_state_out[0],
                                      self.initial_lstm_state1_1 : self.lstm_state_out[1],
                                      self.step_size_1 : [1]} )
    elif game_index == 2:
        v_out, _ = sess.run( [self.v2, self.lstm_state_2],
                         feed_dict = {self.s : [s_t],
                                      self.initial_lstm_state0_2 : self.lstm_state_out[0],
                                      self.initial_lstm_state1_2 : self.lstm_state_out[1],
                                      self.step_size_2 : [1]} )

    else:
        raise ValueError("game index out of range")
    # roll back lstm state
    self.lstm_state_out = prev_lstm_state_out
    return v_out[0]

  def get_vars(self):
    return [self.W_conv1, self.b_conv1,
            self.W_conv2, self.b_conv2,
            self.W_fc1, self.b_fc1,
            self.W_lstm_1, self.b_lstm_1,
            self.W_lstm_2, self.b_lstm_2,
            self.W_fc2_1, self.b_fc2_1,
            self.W_fc2_2, self.b_fc2_2,
            self.W_fc3_1, self.b_fc3_1,
            self.W_fc3_2, self.b_fc3_2]

  def get_vars_1(self):
    return [self.W_conv1, self.b_conv1,
            self.W_conv2, self.b_conv2,
            self.W_fc1, self.b_fc1,
            self.W_lstm_1, self.b_lstm_1,
            self.W_fc2_1, self.b_fc2_1,
            self.W_fc3_1, self.b_fc3_1]


  def get_vars_2(self):
    return [self.W_conv1, self.b_conv1,
            self.W_conv2, self.b_conv2,
            self.W_fc1, self.b_fc1,
            self.W_lstm_2, self.b_lstm_2,
            self.W_fc2_2, self.b_fc2_2,
            self.W_fc3_2, self.b_fc3_2]



