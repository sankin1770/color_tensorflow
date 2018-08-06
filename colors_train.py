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

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and trainvn the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
from datasets import dataset_factory

from datasets.utils import *
import tensorflow as tf
slim = tf.contrib.slim
from preprocessing import preprocessing_factory
import numpy as np
import colors


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'tmp/cifar10_tra',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps',50000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 100,
                            """How often to log results to the console.""")


def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step()

    # Get images and labels for CIFAR-10.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
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
        preprocessing_name = "color" # or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=True)
        image = image_preprocessing_fn(image, 24, 24)

        images, labels = tf.train.shuffle_batch(
            [image, label],
            batch_size=12,
            num_threads=4,
            capacity=2 * 4 * 12,min_after_dequeue=48)

        # labels = slim.one_hot_encoding(labels, 10)
        batch_queue = slim.prefetch_queue.prefetch_queue(
            [images, labels], capacity=16 * 1,
            num_threads=4)

        images, labels = batch_queue.dequeue()

    # with tf.device('/cpu:0'):
    #     img, label = cifar10.read_and_decode("tmp/cifar10_newdata/train.tfrecords")
    #     img_batch, label_batch = tf.train.shuffle_batch([img, label],
    #                                                 batch_size=128, capacity=2000,
    #                                                 min_after_dequeue=1000)
    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = colors.inference(images)
    # logits=cifar10.resnet_50(images, classes=10,is_training=True)
    # model = cifar10_model.ResNetCifar10(
    #     44,
    #     is_training=True,
    #     batch_norm_decay=0.997,
    #     batch_norm_epsilon=1e-5,
    #     data_format='channels_last')

    # logits = model.forward_pass(images, input_data_format='channels_last')
    # logits=cifar10.resnet_50(images)
    # logits=cifar10.resnet_50(images)
    # Calculate loss  and  acc.
    loss,accuracy= colors.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = colors.train(loss, global_step)
    ##### validation step

    # with tf.device('/cpu:0'):
    #     eval_images, eval_labels = cifar10.inputs(eval_data="test")
    # # eval_logits = cifar10.alexnet_cifar_FC(eval_images, True)
    #
    # eval_logits = model.forward_pass(eval_images, input_data_format='channels_last')
    # top_k_op = cifar10.my_accuracy(eval_logits, eval_labels)


    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss,runtime and accuracy."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs([loss,accuracy,logits,labels])  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value,acc_value,logitss,labless= run_values.results
          x=np.argmax(logits)
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f,  batch_accuracy=%.4f   (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,acc_value,
                               examples_per_sec, sec_per_batch))

    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    add_global= global_step.assign_add(1)
    # saver = tf.train.Saver()
    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += bn_moving_vars
    with tf.train.MonitoredTrainingSession(
        save_checkpoint_secs=60,
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               tf.train.SummarySaverHook(save_steps=1000,output_dir=FLAGS.train_dir,summary_op= tf.summary.merge_all()),
               _LoggerHook()],
        config=config) as mon_sess:
        f = open("result.txt", 'a+')
        while not mon_sess.should_stop():
            mon_sess.run(train_op)

            step = mon_sess.run(add_global)
            if step % 1000 == 0:
                lr = mon_sess.run(tf.get_collection('learning_rate'))
                f.write("step %d-----------------------------" % step)
                f.write("lr>>%.5f        " % lr[0])
                # print("%d  learning rate: %f" % (step, lr[0]))
                # eval_acc = 0.0
                # true_count = 0  # Counts the number of correct predictions.
                # total_sample_count = 10000
                # step_1 = 0
                # while step_1 < 156:
                #     predictions = mon_sess.run(top_k_op)
                #     print("%d  eval acc: %f" % (step, eval_acc))
                #     true_count += np.sum(predictions)
                #     step_1 += 1
                #
                #  # Compute precision @ 1.
                # eval_acc = true_count / 10000
                # print("%d  eval acc: %f" % (step, eval_acc))
                # f.write("eval_acc>>%.5f\n" % eval_acc)
                # f.flush()
                #



def main(argv=None):  # pylint: disable=unused-argument

  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()

if __name__ == '__main__':
  tf.app.run()
