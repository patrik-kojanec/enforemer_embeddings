import pandas as pd
import numpy as np
import kipoiseq
from kipoiseq import Interval
import glob
import json
import functools
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import joblib
import gzip
import kipoiseq
from kipoiseq import Interval
import pyfaidx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import sonnet as snt
import sys
import os
import time
from tqdm import tqdm

def rename_stored_variable(v):
    old = v.split("/")
    new = old.copy()
    new[0] = 'enformer'
    new[1] = new[1].replace("_", "")
    
    if (new[1] == 'heads'):
        new[2] = 'head_' + old[2]
        new[3] = new[2]
        new[4] = 'linear'
        new[5] = old[5] + ":" + old[4]
        return("/".join(new[0:6]))
    
    elif (new[1] == 'downres'):
        new[1] = 'trunk/conv_tower/conv_tower'
        new[2] = 'conv_tower_block_' + old[3] + '/' + 'conv_tower_block_' + old[3]
        new[3] = ("conv_block/conv_block", "pointwise_conv_block/pointwise_conv_block", "softmax_pooling/linear/w:0")[int(old[5])]
        if new[5] == '2':
            return("/".join(new[0:4]))
        elif new[5] == '1':    
            new[4] = ("batch_norm", "", "conv1_d")[int(old[8])]
            if int(old[8]) == 0:
                if old[10] in ["scale", "offset"]:
                    new[5] = old[10].replace("_", "",1) + ":0"
                else:
                    new[5] = old[10] + "/" + old[11].replace("_", "",1) + ":0"
                return("/".join(new[0:6]))
            else:
                new[5] = old[9].replace("_", "",1) + ":0"
                return("/".join(new[0:6]))
        else:
            new[4] = ("batch_norm", "", "conv1_d")[int(old[7])]
            if int(old[7]) == 0:
                if old[9] in ["scale", "offset"]:
                    new[5] = old[9].replace("_", "",1) + ":0"
                else:
                    new[5] = old[9] + "/" + old[10].replace("_", "",1) + ":0"
                return("/".join(new[0:6]))
            else:
                new[5] = old[8].replace("_", "",1) + ":0"
                return("/".join(new[0:6]))
                
    elif (new[1] == 'transformer'):
        new[1] = 'trunk/transformer/transformer'
        new[2] = 'transformer_block_' + old[3] + '/' + 'transformer_block_' + old[3]
        new[3] = ('mha/mha', 'mlp/mlp')[int(old[5])]
        if old[5] == '0':                                  # MHA
            if old[10] in ["scale", "offset"]:
                new[4] = "layer_norm"+ "/" + old[10] + ":0"
                return("/".join(new[0:5]))
            else:
                new[4] = "attention_" + old[3]
            if "_bias" in old[10]:
                new[5] = old[10].replace("_","", 1) + ":0"
            elif "embed" in old[10]:
                new[5] = old[10].replace("_","", 1) + "/" + old[11] + ":0"
            else:
                new[5] = old[10].replace("_","", 1) + "/" + old[12] + ":0"
            return("/".join(new[0:6]))
        else:                                              #MLP
            if old[9] in ["scale", "offset"]:
                new[4] = "layer_norm"                 
            else:
                new[4] = "project_out"
            new[5] = old[9].replace("_","", 1)+ ":0"
            return("/".join(new[0:6]))
    
    elif (new[1] == 'stem'):
        new[1] = "trunk/stem/stem"
        if old[3] == '0':
            new[2] = "conv1_d/" + old[4] + ":0"
            return("/".join(new[0:3]))
        elif old[3] == '2':
            return("enformer/trunk/stem/stem/softmax_pooling/linear/w:0")
        else:
            new[2] = "pointwise_conv_block/pointwise_conv_block"
            new[3] = ("batch_norm", "", "conv1_d")[int(old[6])]
            if old[6] == '2':
                new[4] = old[7] + ":0"
                return("/".join(new[0:5]))
            if old[8] in ["scale", "offset"]:
                new[4] = old[8] + ":0"
            else:
                new[4] = old[8] + "/" + old[9].replace("_", "") + ":0"
            return("/".join(new[0:5]))
                
    else:#(new[1] == 'final_pointwise'):  
        new[1] = 'trunk/final_pointwise/final_pointwise'
        new[2] = ("conv_block/conv_block", "pointwise_conv_block/pointwise_conv_block", "softmax_pooling/linear/w:0")[int(old[3])]
        if old[4] == '2':
            return("/".join(new[0:3]))
        new[3] = ("batch_norm", "", "conv1_d")[int(old[5])]
        if old[5] == '2':
            new[4] = old[6] + ":0"
            return("/".join(new[0:5]))
        if old[7] in ["scale", "offset"]:
            new[4] = old[7] + ":0"
        else:
            new[4] = old[7] + "/" + old[8].replace("_", "") + ":0"
        return("/".join(new[0:5]))
            

def one_hot_encode(sequence):
  return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)


##################### Testing utils ###################
# @title `get_targets(organism)`
def get_targets(organism):
  targets_txt = f'https://raw.githubusercontent.com/calico/basenji/master/manuscripts/cross2020/targets_{organism}.txt'
  return pd.read_csv(targets_txt, sep='\t')


def organism_path(organism):
  return os.path.join('gs://basenji_barnyard/data', organism)


def get_dataset(organism, subset, num_threads=8):
  metadata = get_metadata(organism)
  dataset = tf.data.TFRecordDataset(tfrecord_files(organism, subset),
                                    compression_type='ZLIB',
                                    num_parallel_reads=num_threads)
  dataset = tfds.load(tfrecord_files(organism, subset),
                                    num_parallel_reads=num_threads)  
  dataset = dataset.map(functools.partial(deserialize, metadata=metadata),
                        num_parallel_calls=num_threads)
  return dataset


def get_metadata(organism):
  # Keys:
  # num_targets, train_seqs, valid_seqs, test_seqs, seq_length,
  # pool_width, crop_bp, target_length
  if organism == 'mouse':
        return  {'crop_bp': 8192,
                 'num_targets': 1643,
                 'pool_width': 128,
                 'seq_length': 131072,
                 'target_length': 896,
                 'test_seqs': 2017,
                 'train_seq': 29295,
                 'valid_seqs': 2209}
  else:
        return {'crop_bp': 8192,
                'num_targets': 5313,
                'pool_width': 128,
                'seq_length': 131072,
                'target_length': 896,
                'test_seqs': 1937,
                'train_seq': 34021,
                'valid_seqs': 2213}




def tfrecord_files(organism, subset):
  # Sort the values by int(*).
  return sorted(tf.io.gfile.glob(os.path.join(
      organism_path(organism), 'tfrecords', f'{subset}-*.tfr'
  )), key=lambda x: int(x.split('-')[-1].split('.')[0]))


def deserialize(serialized_example, metadata):
  """Deserialize bytes stored in TFRecordFile."""
  feature_map = {
      'sequence': tf.io.FixedLenFeature([], tf.string),
      'target': tf.io.FixedLenFeature([], tf.string),
  }
  example = tf.io.parse_example(serialized_example, feature_map)
  sequence = tf.io.decode_raw(example['sequence'], tf.bool)
  sequence = tf.reshape(sequence, (metadata['seq_length'], 4))
  sequence = tf.cast(sequence, tf.float32)

  target = tf.io.decode_raw(example['target'], tf.float16)
  target = tf.reshape(target,
                      (metadata['target_length'], metadata['num_targets']))
  target = tf.cast(target, tf.float32)

  return {'sequence': sequence,
          'target': target}
          
def _reduced_shape(shape, axis):
  if axis is None:
    return tf.TensorShape([])
  return tf.TensorShape([d for i, d in enumerate(shape) if i not in axis])


class CorrelationStats(tf.keras.metrics.Metric):
  """Contains shared code for PearsonR and R2."""

  def __init__(self, reduce_axis=None, name='pearsonr'):
    """Pearson correlation coefficient.

    Args:
      reduce_axis: Specifies over which axis to compute the correlation (say
        (0, 1). If not specified, it will compute the correlation across the
        whole tensor.
      name: Metric name.
    """
    super(CorrelationStats, self).__init__(name=name)
    self._reduce_axis = reduce_axis
    self._shape = None  # Specified in _initialize.

  def _initialize(self, input_shape):
    # Remaining dimensions after reducing over self._reduce_axis.
    self._shape = _reduced_shape(input_shape, self._reduce_axis)

    weight_kwargs = dict(shape=self._shape, initializer='zeros')
    self._count = self.add_weight(name='count', **weight_kwargs)
    self._product_sum = self.add_weight(name='product_sum', **weight_kwargs)
    self._true_sum = self.add_weight(name='true_sum', **weight_kwargs)
    self._true_squared_sum = self.add_weight(name='true_squared_sum',
                                             **weight_kwargs)
    self._pred_sum = self.add_weight(name='pred_sum', **weight_kwargs)
    self._pred_squared_sum = self.add_weight(name='pred_squared_sum',
                                             **weight_kwargs)

  def update_state(self, y_true, y_pred, sample_weight=None):
    """Update the metric state.

    Args:
      y_true: Multi-dimensional float tensor [batch, ...] containing the ground
        truth values.
      y_pred: float tensor with the same shape as y_true containing predicted
        values.
      sample_weight: 1D tensor aligned with y_true batch dimension specifying
        the weight of individual observations.
    """
    if self._shape is None:
      # Explicit initialization check.
      self._initialize(y_true.shape)
    y_true.shape.assert_is_compatible_with(y_pred.shape)
    y_true = tf.cast(y_true, 'float32')
    y_pred = tf.cast(y_pred, 'float32')

    self._product_sum.assign_add(
        tf.reduce_sum(y_true * y_pred, axis=self._reduce_axis))

    self._true_sum.assign_add(
        tf.reduce_sum(y_true, axis=self._reduce_axis))

    self._true_squared_sum.assign_add(
        tf.reduce_sum(tf.math.square(y_true), axis=self._reduce_axis))

    self._pred_sum.assign_add(
        tf.reduce_sum(y_pred, axis=self._reduce_axis))

    self._pred_squared_sum.assign_add(
        tf.reduce_sum(tf.math.square(y_pred), axis=self._reduce_axis))

    self._count.assign_add(
        tf.reduce_sum(tf.ones_like(y_true), axis=self._reduce_axis))

  def result(self):
    raise NotImplementedError('Must be implemented in subclasses.')

  def reset_states(self):
    if self._shape is not None:
      tf.keras.backend.batch_set_value([(v, np.zeros(self._shape))
                                        for v in self.variables])


class PearsonR(CorrelationStats):
  """Pearson correlation coefficient.

  Computed as:
  ((x - x_avg) * (y - y_avg) / sqrt(Var[x] * Var[y])
  """

  def __init__(self, reduce_axis=(0,), name='pearsonr'):
    """Pearson correlation coefficient.

    Args:
      reduce_axis: Specifies over which axis to compute the correlation.
      name: Metric name.
    """
    super(PearsonR, self).__init__(reduce_axis=reduce_axis,
                                   name=name)

  def result(self):
    true_mean = self._true_sum / self._count
    pred_mean = self._pred_sum / self._count

    covariance = (self._product_sum
                  - true_mean * self._pred_sum
                  - pred_mean * self._true_sum
                  + self._count * true_mean * pred_mean)

    true_var = self._true_squared_sum - self._count * tf.math.square(true_mean)
    pred_var = self._pred_squared_sum - self._count * tf.math.square(pred_mean)
    tp_var = tf.math.sqrt(true_var) * tf.math.sqrt(pred_var)
    correlation = covariance / tp_var

    return correlation


class R2(CorrelationStats):
  """R-squared  (fraction of explained variance)."""

  def __init__(self, reduce_axis=None, name='R2'):
    """R-squared metric.

    Args:
      reduce_axis: Specifies over which axis to compute the correlation.
      name: Metric name.
    """
    super(R2, self).__init__(reduce_axis=reduce_axis,
                             name=name)

  def result(self):
    true_mean = self._true_sum / self._count
    total = self._true_squared_sum - self._count * tf.math.square(true_mean)
    residuals = (self._pred_squared_sum - 2 * self._product_sum
                 + self._true_squared_sum)

    return tf.ones_like(residuals) - residuals / total


class MetricDict:
  def __init__(self, metrics):
    self._metrics = metrics

  def update_state(self, y_true, y_pred):
    for k, metric in self._metrics.items():
      metric.update_state(y_true, y_pred)

  def result(self):
    return {k: metric.result() for k, metric in self._metrics.items()}
    
