# -*- coding: utf-8 -*-
import tensorflow as tf
import threading
import numpy as np

import datetime
import signal
import random
import math
import os
import time

from game_ac_network_cotrain import GameACFFNetwork, GameACLSTMNetwork
from a3c_training_thread_cotrain import A3CTrainingThread
from rmsprop_applier import RMSPropApplier

from constants_cotrain import ACTION_SIZE1, ACTION_SIZE2
from constants_cotrain import PARALLEL_SIZE
from constants_cotrain import INITIAL_ALPHA_LOW
from constants_cotrain import INITIAL_ALPHA_HIGH
from constants_cotrain import INITIAL_ALPHA_LOG_RATE
from constants_cotrain import MAX_TIME_STEP
from constants_cotrain import CHECKPOINT_DIR
from constants_cotrain import LOG_FILE
from constants_cotrain import RMSP_EPSILON
from constants_cotrain import RMSP_ALPHA
from constants_cotrain import GRAD_NORM_CLIP
from constants_cotrain import USE_GPU
from constants_cotrain import USE_LSTM
from constants_cotrain import ROM1, ROM2


def log_uniform(lo, hi, rate):
  log_lo = math.log(lo)
  log_hi = math.log(hi)
  v = log_lo * (1-rate) + log_hi * rate
  return math.exp(v)

device = "/cpu:0"
if USE_GPU:
  device = "/gpu:0"

initial_learning_rate = log_uniform(INITIAL_ALPHA_LOW,
                                    INITIAL_ALPHA_HIGH,
                                    INITIAL_ALPHA_LOG_RATE)

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
ENV1 = os.path.basename(ROM1).rstrip('.bin')
ENV2 = os.path.basename(ROM2).rstrip('.bin')
CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, ENV1 + ENV2 + st)

global_t = 0

stop_requested = False

if USE_LSTM:
  global_network = GameACLSTMNetwork(ACTION_SIZE1, -1, device)
else:
  #global_network needs to incorporate two games
  global_network = GameACFFNetwork(ACTION_SIZE1, ACTION_SIZE2, -1, device)


training_threads = []

learning_rate_input = tf.placeholder("float")

grad_applier = RMSPropApplier(learning_rate = learning_rate_input,
                              decay = RMSP_ALPHA,
                              momentum = 0.0,
                              epsilon = RMSP_EPSILON,
                              clip_norm = GRAD_NORM_CLIP,
                              device = device)

for i in range(PARALLEL_SIZE):
  #each training thread is a A3CTraininingThread object
  training_thread = A3CTrainingThread(i, global_network, initial_learning_rate,
                                      learning_rate_input,
                                      grad_applier, MAX_TIME_STEP,
                                      device = device)
  training_threads.append(training_thread)

# prepare session
#config = tf.ConfigProto()
#config.gpu_options.allow_growth=True
#config.log_device_placement = False
#config.allow_soft_placement = True
#sess = tf.Session(config = config)
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                        allow_soft_placement=True,
                                        gpu_options=gpu_options))

init = tf.global_variables_initializer()
sess.run(init)

# summary for tensorboard
# one score writer for each game
score_input1 = tf.placeholder(tf.int32)
score_input2 = tf.placeholder(tf.int32)
summary_op1 = tf.summary.scalar("score1", score_input1)
summary_op2 = tf.summary.scalar("score2", score_input2)

#summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(LOG_FILE, sess.graph)

# init or load checkpoint with saver
saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
if checkpoint and checkpoint.model_checkpoint_path:
  saver.restore(sess, checkpoint.model_checkpoint_path)
  print("checkpoint loaded:", checkpoint.model_checkpoint_path)
  tokens = checkpoint.model_checkpoint_path.split("-")
  # set global step
  global_t = int(tokens[1])
  print(">>> global step set: ", global_t)
  # set wall time
  wall_t_fname = CHECKPOINT_DIR + '/' + 'wall_t.' + str(global_t)
  with open(wall_t_fname, 'r') as f:
    wall_t = float(f.read())
else:
  print("Could not find old checkpoint")
  # set wall time
  wall_t = 0.0


def train_function(parallel_index):
  global global_t
  
  training_thread = training_threads[parallel_index]
  # set start_time
  start_time = time.time() - wall_t
  training_thread.set_start_time(start_time)

  while True:
    if stop_requested:
      break
    if global_t > MAX_TIME_STEP:
      break

    if parallel_index % 2 == 0:
        diff_global_t = training_thread.process(sess, global_t, summary_writer, summary_op1, score_input1)
    elif parallel_index % 2 == 1:
        diff_global_t = training_thread.process(sess, global_t, summary_writer, summary_op2, score_input2)
    else:
        raise ValueError("parallel_index should be an integer")
    global_t += diff_global_t
    
    
def signal_handler(signal, frame):
  global stop_requested
  print('You pressed Ctrl+C!')
  stop_requested = True
  
train_threads = []
for i in range(PARALLEL_SIZE):
  train_threads.append(threading.Thread(target=train_function, args=(i,)))
  
signal.signal(signal.SIGINT, signal_handler)

# set start time
start_time = time.time() - wall_t

for t in train_threads:
  t.start()

print('Press Ctrl+C to stop')
signal.pause()

print('Now saving data. Please wait')
  
for t in train_threads:
  t.join()

if not os.path.exists(CHECKPOINT_DIR):
  os.mkdir(CHECKPOINT_DIR)  

# write wall time
wall_t = time.time() - start_time
wall_t_fname = CHECKPOINT_DIR + '/' + 'wall_t.' + str(global_t)
with open(wall_t_fname, 'w') as f:
  f.write(str(wall_t))

saver.save(sess, CHECKPOINT_DIR + '/' + 'checkpoint', global_step = global_t)

