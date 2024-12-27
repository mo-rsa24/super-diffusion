import jax
import numpy as np
import tensorflow as tf
import scipy

def get_inception_model():
  model = tf.keras.applications.InceptionV3(
    include_top=False,
    weights='imagenet',
    input_shape=(299, 299, 3),
    pooling='avg')
  return model
  
@tf.function
def run_inception_jit(inputs, model):
  inputs = tf.cast(inputs, tf.float32)
  inputs = tf.image.resize(inputs, size=(299, 299))
  inputs = tf.keras.applications.inception_v3.preprocess_input(inputs)
  return model(inputs)
  
@tf.function
def run_inception_distributed(input_tensor, model):
  num_tpus = jax.local_device_count()
  input_tensors = tf.split(input_tensor, num_tpus, axis=0)
  pool3 = []
  device_format = '/TPU:{}' if 'TPU' in str(jax.devices()[0]) else '/GPU:{}'
  for i, tensor in enumerate(input_tensors):
    with tf.device(device_format.format(i)):
      tensor_on_device = tf.identity(tensor)
      pool3.append(run_inception_jit(tensor_on_device, model))

  with tf.device('/CPU'):
    return tf.concat(pool3, axis=0)

def fid(X, Y, eps=1e-6):
    assert X.shape[1] == Y.shape[1]
    mu_x, mu_y = np.mean(X, axis=0), np.mean(Y, axis=0)
    s_x, s_y = np.cov(X, rowvar=False), np.cov(Y, rowvar=False)
    out = ((mu_x - mu_y)**2).sum()
    sqr = scipy.linalg.sqrtm(s_x@s_y)
    if not np.isfinite(sqr).all():
        print(f'regularizing the matrix with eps={eps}') 
        sqr += eps*np.eye(sqr.shape[0])
    out += np.trace(s_x + s_y - 2*sqr)
    return out
  
def load_dataset_stats(config, eval=False):
  """Load the pre-computed dataset statistics."""
  suffix = 'test' if eval else 'train'
  if config.data.dataset == 'CIFAR10':
    filename = f'assets/stats/cifar10_{suffix}_stats.npz'
  else:
    raise ValueError(f'Dataset {config.data.dataset} stats not found.')

  with tf.io.gfile.GFile(filename, 'rb') as fin:
    stats = np.load(fin)
    return stats
