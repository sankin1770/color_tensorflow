# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the CIFAR-10 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf



FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 64,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', 'tmp/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Global constants describing the CIFAR-10 data set.

NUM_CLASSES = 8



# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.001  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var



def inference(images):
  """Build the CIFAR-10 model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 3, 64],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 64, 64],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv2)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [images.get_shape().as_list()[0], -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    _activation_summary(local4)

  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=None)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear


def conv2d(_x, _w, _b):
    return tf.nn.bias_add(tf.nn.conv2d(_x, _w, [1, 1, 1, 1], padding='SAME'), _b)


def max_pool(_x, f):
    return tf.nn.max_pool(_x, [1, f, f, 1], [1, 1, 1, 1], padding='SAME')


def lrn(_x):
    return tf.nn.lrn(_x, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)


def init_w(namespace, shape, wd, stddev, reuse=False):
    with tf.variable_scope(namespace, reuse=reuse):
        initializer = tf.truncated_normal_initializer(dtype=tf.float32, stddev=stddev)
        w = tf.get_variable("w", shape=shape, initializer=initializer)

        if wd:
            weight_decay = tf.multiply(tf.nn.l2_loss(w), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
    return w


def init_b(namespace, shape, reuse=False):
    with tf.variable_scope(namespace, reuse=reuse):
        initializer = tf.constant_initializer(0.0)
        b = tf.get_variable("b", shape=shape, initializer=initializer)
    return b


def batch_normal(xs, out_size):
    axis = list(range(len(xs.get_shape()) - 1))
    n_mean, n_var = tf.nn.moments(xs, axes=axis)
    scale = tf.Variable(tf.ones([out_size]))
    shift = tf.Variable(tf.zeros([out_size]))
    epsilon = 0.001
    ema = tf.train.ExponentialMovingAverage(decay=0.9)

    def mean_var_with_update():
        ema_apply_op = ema.apply([n_mean, n_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(n_mean), tf.identity(n_var)

    mean, var = mean_var_with_update()

    bn = tf.nn.batch_normalization(xs, mean, var, shift, scale, epsilon)
    return bn


def alexnet_cifar(images, reuse=False):
    '''Build the network model and return logits'''

    # conv1
    w1 = init_w("conv1", [3, 3, 3, 24], None, 0.01, reuse)
    bw1 = init_b("conv1", [24], reuse)
    conv1 = conv2d(images, w1, bw1)
    bn1 = batch_normal(conv1, 24)
    c_output1 = tf.nn.relu(bn1)
    pool1 = max_pool(c_output1, 2)

    # conv2
    w2 = init_w("conv2", [3, 3, 24, 96], None, 0.01, reuse)
    bw2 = init_b("conv2", [96], reuse)
    conv2 = conv2d(pool1, w2, bw2)
    bn2 = batch_normal(conv2, 96)
    c_output2 = tf.nn.relu(bn2)
    pool2 = max_pool(c_output2, 2)

    # conv3
    w3 = init_w("conv3", [3, 3, 96, 192], None, 0.01, reuse)
    bw3 = init_b("conv3", [192], reuse)
    conv3 = conv2d(pool2, w3, bw3)
    bn3 = batch_normal(conv3, 192)
    c_output3 = tf.nn.relu(bn3)

    # conv4
    w4 = init_w("conv4", [3, 3, 192, 192], None, 0.01, reuse)
    bw4 = init_b("conv4", [192], reuse)
    conv4 = conv2d(conv3, w4, bw4)
    bn4 = batch_normal(conv4, 192)
    c_output4 = tf.nn.relu(bn4)

    # conv5
    w5 = init_w("conv5", [3, 3, 192, 96], None, 0.01, reuse)
    bw5 = init_b("conv5", [96], reuse)
    conv5 = conv2d(conv4, w5, bw5)
    bn5 = batch_normal(conv5, 96)
    c_output5 = tf.nn.relu(bn5)
    pool5 = max_pool(c_output5, 2)
    print(c_output5.shape)
    # FC1
    wfc1 = init_w("fc1", [96 * 24 * 24, 1024], None, 1e-2, reuse)
    bfc1 = init_b("fc1", [1024], reuse)
    shape = pool5.get_shape()
    reshape = tf.reshape(pool5, [-1, shape[1].value * shape[2].value * shape[3].value])
    w_x1 = tf.matmul(reshape, wfc1) + bfc1
    bn6 = batch_normal(w_x1, 1024)
    fc1 = tf.nn.relu(bn6)

    # FC2
    wfc2 = init_w("fc2", [1024, 1024], None, 1e-2, reuse)
    bfc2 = init_b("fc2", [1024], reuse)
    w_x2 = tf.matmul(fc1, wfc2) + bfc2
    bn7 = batch_normal(w_x2, 1024)
    fc2 = tf.nn.relu(bn7)

    # FC3
    wfc3 = init_w("fc3", [1024, 10], None, 1e-2, reuse)
    bfc3 = init_b("fc3", [10], reuse)
    softmax_linear = tf.add(tf.matmul(fc2, wfc3), bfc3)


    return softmax_linear

def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)

  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  # logits=tf.nn.softmax(logits)
  predictions=tf.argmax(logits,axis=1)
  acc= tf.reduce_mean(tf.to_float(tf.equal(predictions,labels)))
  tf.add_to_collection('losses', cross_entropy_mean)
  tf.summary.scalar('train_acc', acc)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss'),acc

def my_accuracy(logits, labels):
    labels = tf.cast(labels, tf.int64)
    predictions = tf.argmax(logits, axis=1)
    acc = tf.reduce_mean(tf.to_float(tf.equal(predictions, labels)))
    return acc
def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = 7799 / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)
  tf.add_to_collection('learning_rate', lr)
  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies([loss_averages_op]):
      with tf.control_dependencies(update_ops):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  with tf.control_dependencies([apply_gradient_op]):
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

  return variables_averages_op

