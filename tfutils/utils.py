"""

"""
import tensorflow as tf
import os

from training_utils import yaml_load


def ndim(x):
    """Returns the number of axes in a tensor, as an integer.
    """
    return len(x.get_shape())


def binary_accuracy(y_true, y_pred):
    return tf.reduce_mean(tf.equal(y_true, tf.round(y_pred)), axis=-1)


def categorical_accuracy(y_true, y_pred, mean=True):
    acc = tf.cast(tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1)),
                  tf.float32)
    if mean:
        acc = tf.reduce_mean(acc, axis=-1)

    return acc


def sparse_categorical_accuracy(y_true, y_pred, mean=True):
    # reshape in case it's in shape (num_samples, 1) instead of (num_samples,)
    if ndim(y_true) == ndim(y_pred):
        y_true = tf.squeeze(y_true, -1)
    # convert dense predictions to labels
    y_pred_labels = tf.argmax(y_pred, axis=-1)
    y_pred_labels = tf.cast(y_pred_labels, y_true.dtype)
    acc = tf.cast(tf.equal(y_true, y_pred_labels), tf.float32)
    if mean:
        acc = tf.reduce_mean(acc, axis=-1)
    return acc


def top_k_categorical_accuracy(y_true, y_pred, k=5):
    return tf.reduce_mean(tf.nn.in_top_k(y_pred, tf.argmax(y_true, axis=-1), k), axis=-1)


def sparse_top_k_categorical_accuracy(y_true, y_pred, k=5):
    # If the shape of y_true is (num_samples, 1), flatten to (num_samples,)
    return tf.reduce_mean(tf.nn.in_top_k(y_pred, tf.cast(tf.reshape(y_true, [-1]), 'int32'), k), axis=-1)


#####################################################################
# tf operations
#####################################################################


def is_trainable(var):
    """Whether var is in TRAINABLE_VARIABLES."""
    return var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)


def restore_latest_checkpoint(saver, sess, ckpts_dir):
    p = yaml_load(os.path.join(ckpts_dir, 'checkpoint'))['model_checkpoint_path']
    if not p.startswith('/'):
        p = os.path.join(ckpts_dir, p)
    saver.restore(sess, p)
