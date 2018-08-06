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

"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datasets import dataset_factory
from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import colors

from preprocessing import preprocessing_factory
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', 'tmp/cifar10_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'tmp/cifar10_tra',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 7799,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")


def eval_once(saver, summary_writer, top_k_op,top_k_op_5, summary_op,logits):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / 12))
      true_count = 0  # Counts the number of correct predictions.
      true_count_5 = 0
      total_sample_count = num_iter * 12
      step = 0
      while step < num_iter and not coord.should_stop():
        predictions,predictions_5 ,tt= sess.run([top_k_op,top_k_op_5,logits])
        true_count_5 += np.sum(predictions_5)
        true_count += np.sum(predictions)
        step += 1

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      precision5 = true_count_5 / total_sample_count
      # print('  top5 = %.3f%' % precision5)
      print('"{}    test_acc:{:.2%}   top5:{:.2%}"'.format(datetime.now(), precision, precision5))
      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      # summary.value.add(tag='Precision @ 5', simple_value=precision5)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    # images, labels = cifar10.inputs(eval_data="test")
    dataset = dataset_factory.get_dataset(
      "color", "train", "D:/colors")
    examples_per_shard = 1024
    min_queue_examples = examples_per_shard * 50
    provider = slim.dataset_data_provider.DatasetDataProvider(
      dataset,
      num_readers=8,
      common_queue_capacity=min_queue_examples + 3 * 12,
      common_queue_min=min_queue_examples)
    [image, label] = provider.get(['image', 'label'])
    # image,label=set(image,label_1,label_2,FLAGS.coarse,fw[FLAGS.coarse])
    preprocessing_name = "color"  # or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
      preprocessing_name,
      is_training=False)
    image = image_preprocessing_fn(image, 24, 24)

    images, labels = tf.train.batch(
      [image, label],
      batch_size=12,
      num_threads=4,
      capacity=2 * 4 * 12)

    # labels = slim.one_hot_encoding(labels, 10)
    batch_queue = slim.prefetch_queue.prefetch_queue(
      [images, labels], capacity=16 * 1,
      num_threads=4)

    images, labels = batch_queue.dequeue()
    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits=colors.inference(images)

    # logits = resnet_my.resnet50(images,training=False)
    # model = cifar10_model.ResNetCifar10(
    #     44,
    #     is_training=False,
    #     batch_norm_decay=0.997,
    #     batch_norm_epsilon=1e-5,
    #     data_format='channels_last')
    # logits = model.forward_pass(images, input_data_format='channels_last')

    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)
    top_k_op_5 = tf.nn.in_top_k(logits, labels, 5)
    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        colors.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    while True:
      eval_once(saver, summary_writer, top_k_op,top_k_op_5, summary_op,logits)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
