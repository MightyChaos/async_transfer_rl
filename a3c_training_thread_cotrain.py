# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import time
import sys

from game_state_cotrain import GameState1, GameState2
from game_state_cotrain import ACTION_SIZE1, ACTION_SIZE2
from game_ac_network_cotrain import GameACFFNetwork, GameACLSTMNetwork

from constants_cotrain import GAMMA
from constants_cotrain import LOCAL_T_MAX
from constants_cotrain import ENTROPY_BETA
from constants_cotrain import USE_LSTM

LOG_INTERVAL = 100
PERFORMANCE_LOG_INTERVAL = 1000

class A3CTrainingThread(object):
  def __init__(self,
               thread_index,
               global_network,
               initial_learning_rate,
               learning_rate_input,
               grad_applier,
               max_global_time_step,
               device):
    self.thread_index = thread_index
    self.learning_rate_input = learning_rate_input
    self.max_global_time_step = max_global_time_step

    if USE_LSTM:
      self.local_network = GameACLSTMNetwork(ACTION_SIZE1, thread_index, device)
    else:
      self.local_network = GameACFFNetwork(ACTION_SIZE1, ACTION_SIZE2, thread_index, device)

    self.local_network.prepare_loss(ENTROPY_BETA)

    if thread_index % 2 == 0:
      self.game_index = 1
      self.ACTION_SIZE = ACTION_SIZE1
      self.GameState = GameState1
      self.loss = self.local_network.total_loss1
    else:
      self.game_index = 2
      self.ACTION_SIZE = ACTION_SIZE2
      self.GameState = GameState2
      self.loss = self.local_network.total_loss2

    with tf.device(device):
      if thread_index % 2 == 0:
        var_refs = [v._ref() for v in self.local_network.get_vars_1()]
      elif thread_index % 2 == 1:
        var_refs = [v._ref() for v in self.local_network.get_vars_2()]
      else:
        raise ValueError("thread index should be integer")
      self.gradients = tf.gradients(
        self.loss, var_refs,
        gate_gradients=False,
        aggregation_method=None,
        colocate_gradients_with_ops=False)

    grad_shape = self.gradients
    print("grad_shape {0}".format(grad_shape))
    if thread_index % 2 == 0:
        self.apply_gradients = grad_applier.apply_gradients(
          global_network.get_vars_1(),
          self.gradients)
    elif thread_index % 2 == 1:
        self.apply_gradients = grad_applier.apply_gradients(
          global_network.get_vars_2(),
          self.gradients )
      
    self.sync = self.local_network.sync_from(global_network)
    
    self.game_state = self.GameState(113 * thread_index)
    
    self.local_t = 0

    self.initial_learning_rate = initial_learning_rate

    self.episode_reward = 0

    # variable controling log output
    self.prev_local_t = 0

  def _anneal_learning_rate(self, global_time_step):
    learning_rate = self.initial_learning_rate * (self.max_global_time_step - global_time_step) / self.max_global_time_step
    if learning_rate < 0.0:
      learning_rate = 0.0
    return learning_rate

  def choose_action(self, pi_values):
    return np.random.choice(range(len(pi_values)), p=pi_values)

  def _record_score(self, sess, summary_writer, summary_op, score_input, score, global_t):
    summary_str = sess.run(summary_op, feed_dict={
      score_input: score
    })
    summary_writer.add_summary(summary_str, global_t)
    summary_writer.flush()
    
  def set_start_time(self, start_time):
    self.start_time = start_time

  def process(self, sess, global_t, summary_writer, summary_op, score_input):
    states = []
    actions = []
    rewards = []
    values = []

    terminal_end = False

    # copy weights from shared to local
    sess.run( self.sync )

    start_local_t = self.local_t

    if USE_LSTM:
      start_lstm_state = self.local_network.lstm_state_out
    
    # t_max times loop
    for i in range(LOCAL_T_MAX):
      pi_, value_ = self.local_network.run_policy_and_value(sess, self.game_state.s_t, self.game_index)
      action = self.choose_action(pi_)

      states.append(self.game_state.s_t)
      actions.append(action)
      values.append(value_)

      if (self.thread_index == 0) and (self.local_t % LOG_INTERVAL == 0):
        print("iter={0}, game={1}, pi={2}, V={3}".format(i, self.game_index, pi_, value_))
        #print(" V={}".format(value_))

      # process game
      self.game_state.process(action)

      # receive game result
      reward = self.game_state.reward
      terminal = self.game_state.terminal

      self.episode_reward += reward

      # clip reward
      rewards.append( np.clip(reward, -1, 1) )

      self.local_t += 1

      # s_t1 -> s_t
      self.game_state.update()
      
      if terminal:
        terminal_end = True
        print("game {} score={}".format(self.game_index, self.episode_reward))

        self._record_score(sess, summary_writer, summary_op, score_input,
                           self.episode_reward, global_t)
          
        self.episode_reward = 0
        self.game_state.reset()
        if USE_LSTM:
          self.local_network.reset_state()
        break

    R = 0.0
    if not terminal_end:
      R = self.local_network.run_value(sess, self.game_state.s_t, self.game_index)

    actions.reverse()
    states.reverse()
    rewards.reverse()
    values.reverse()

    batch_si = []
    batch_a = []
    batch_td = []
    batch_R = []

    # compute and accmulate gradients
    for(ai, ri, si, Vi) in zip(actions, rewards, states, values):
      R = ri + GAMMA * R
      td = R - Vi
      a = np.zeros([self.ACTION_SIZE])
      a[ai] = 1

      batch_si.append(si)
      batch_a.append(a)
      batch_td.append(td)
      batch_R.append(R)

    cur_learning_rate = self._anneal_learning_rate(global_t)

    if USE_LSTM:
      batch_si.reverse()
      batch_a.reverse()
      batch_td.reverse()
      batch_R.reverse()

      sess.run( self.apply_gradients,
                feed_dict = {
                  self.local_network.s: batch_si,
                  self.local_network.a: batch_a,
                  self.local_network.td: batch_td,
                  self.local_network.r: batch_R,
                  self.local_network.initial_lstm_state: start_lstm_state,
                  self.local_network.step_size : [len(batch_a)],
                  self.learning_rate_input: cur_learning_rate } )
    else:
      if self.game_index == 1:
          sess.run( self.apply_gradients,
                    feed_dict = {
                      self.local_network.s: batch_si,
                      self.local_network.a1: batch_a,
                      self.local_network.td1: batch_td,
                      self.local_network.r1: batch_R,
                      self.learning_rate_input: cur_learning_rate} )
      elif self.game_index == 2:
           sess.run( self.apply_gradients,
                    feed_dict = {
                      self.local_network.s: batch_si,
                      self.local_network.a2: batch_a,
                      self.local_network.td2: batch_td,
                      self.local_network.r2: batch_R,
                      self.learning_rate_input: cur_learning_rate} )

    if (self.thread_index == 0) and (self.local_t - self.prev_local_t >= PERFORMANCE_LOG_INTERVAL):
      self.prev_local_t += PERFORMANCE_LOG_INTERVAL
      elapsed_time = time.time() - self.start_time
      steps_per_sec = global_t / elapsed_time
      print("### Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(
        global_t,  elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))

    # return advanced local step size
    diff_local_t = self.local_t - start_local_t
    return diff_local_t
    