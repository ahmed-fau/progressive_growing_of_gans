import cPickle as pickle
import os
import sys

import numpy as np
import tensorflow as tf

ckpt_fp, out_dir = sys.argv[1:]

_step = int(os.path.basename(ckpt_fp).split('_')[0])
print _step

with open(ckpt_fp, 'rb') as f:
  g_vars = {n: v for n, v in pickle.load(f)}

def get_np_var_by_name(vn):
  return g_vars[vn]

g_init = {}
def make_tf_var(v, name, trainable=True, dtype=tf.float32):
  tf_var = tf.get_variable(name, v.shape, dtype=tf.float32)
  g_init[name] = (tf_var, v)
  return tf_var

def pixel_norm(x):
  # NOTE: axis=-1 because we're using NWC rather than NCW
  return x / tf.sqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + 1e-8)

def wscale(x, scale, vn, nonlinearity):
  scale = make_tf_var(np.array(scale, dtype=np.float32), vn.replace('.b', '.scale'), trainable=False)
  return nonlinearity((x * scale) + make_tf_var(get_np_var_by_name(vn), vn))

def upscale(x, scale=4):
  _, w, nch = x.get_shape().as_list()
  
  x = tf.expand_dims(x, axis=1)
  x = tf.image.resize_nearest_neighbor(x, [1, w * scale])
  x = x[:, 0]
  
  return x

conv_filter = lambda vn: make_tf_var(np.transpose(get_np_var_by_name(vn), [2, 1, 0])[::-1], vn)

samp_n = tf.placeholder(tf.int32, [], name='samp_n')
samp_z = tf.random_normal([samp_n, 512], name='samp_z')

z = tf.placeholder(tf.float32, [None, 512], name='z')

z_norm = pixel_norm(z)

x = tf.expand_dims(z_norm, axis=1)

with tf.variable_scope('G'):
  # Conv1 (projects latents)
  x = tf.pad(x, [[0, 0], [15, 15], [0, 0]])
  x = tf.nn.conv1d(x, conv_filter('G1a.W'), 1, padding='VALID', data_format='NWC')
  x = wscale(x, 0.015617936, 'G1aS.b', tf.nn.leaky_relu)
  x = tf.nn.conv1d(x, conv_filter('G1b.W'), 1, padding='SAME', data_format='NWC')
  x = wscale(x, 0.020833442, 'G1bS.b', tf.nn.leaky_relu)

  # Conv2
  x = upscale(x)
  x = tf.nn.conv1d(x, conv_filter('G2a.W'), 1, padding='SAME', data_format='NWC')
  x = wscale(x, 0.020837622, 'G2aS.b', tf.nn.leaky_relu)
  x = tf.nn.conv1d(x, conv_filter('G2b.W'), 1, padding='SAME', data_format='NWC')
  x = wscale(x, 0.020843236, 'G2bS.b', tf.nn.leaky_relu)

  # Conv3
  x = upscale(x)
  x = tf.nn.conv1d(x, conv_filter('G3a.W'), 1, padding='SAME', data_format='NWC')
  x = wscale(x, 0.020820292, 'G3aS.b', tf.nn.leaky_relu)
  x = tf.nn.conv1d(x, conv_filter('G3b.W'), 1, padding='SAME', data_format='NWC')
  x = wscale(x, 0.020836806, 'G3bS.b', tf.nn.leaky_relu)

  # Conv4
  x = upscale(x)
  x = tf.nn.conv1d(x, conv_filter('G4a.W'), 1, padding='SAME', data_format='NWC')
  x = wscale(x, 0.02082933, 'G4aS.b', tf.nn.leaky_relu)
  x = tf.nn.conv1d(x, conv_filter('G4b.W'), 1, padding='SAME', data_format='NWC')
  x = wscale(x, 0.020813614, 'G4bS.b', tf.nn.leaky_relu)

  # Conv5
  x = upscale(x)
  x = tf.nn.conv1d(x, conv_filter('G5a.W'), 1, padding='SAME', data_format='NWC')
  x = wscale(x, 0.020801041, 'G5aS.b', tf.nn.leaky_relu)
  x = tf.nn.conv1d(x, conv_filter('G5b.W'), 1, padding='SAME', data_format='NWC')
  x = wscale(x, 0.0294875, 'G5bS.b', tf.nn.leaky_relu)

  # Conv6
  x = upscale(x)
  x = tf.nn.conv1d(x, conv_filter('G6a.W'), 1, padding='SAME', data_format='NWC')
  x = wscale(x, 0.029462723, 'G6aS.b', tf.nn.leaky_relu)
  x = tf.nn.conv1d(x, conv_filter('G6b.W'), 1, padding='SAME', data_format='NWC')
  x = wscale(x, 0.04161643, 'G6bS.b', tf.nn.leaky_relu)

  # Aggregate
  f = make_tf_var(np.reshape(get_np_var_by_name('Glod0.W'), [1, 128, 1]), 'Glob0.W')
  x = tf.nn.conv1d(x, f, 1, padding='VALID', data_format='NWC')
  x = wscale(x, 0.08200226, 'Glod0S.b', tf.identity)

  Gz = tf.identity(x, name='Gz')

G_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='G')
step = tf.Variable(_step, dtype=tf.int64, name='global_step')
saver = tf.train.Saver(G_vars + [step])

for var in G_vars:
  print var.name, var.get_shape()

metagraph_fp = os.path.join(out_dir, 'generate.meta')
tf.train.export_meta_graph(
    filename=metagraph_fp,
    clear_devices=True,
    saver_def=saver.as_saver_def())

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for name, (tf_var, np_var) in g_init.items():
    sess.run(tf.assign(tf_var, np_var))

  ckpt_fp = os.path.join(out_dir, 'model.ckpt')
  saver.save(sess, ckpt_fp, global_step=_step, write_meta_graph=False)
