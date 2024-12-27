import gc
import os
import functools
import json
import io

import wandb
from tqdm.auto import tqdm, trange

import jax
import orbax
import flax
import numpy as np
import tensorflow as tf
import flax.jax_utils as flax_utils
from jax import random, jit
from jax import numpy as jnp
from flax.training import checkpoints
import math

import evaluation

import datasets
import train_utils as tutils
import eval_utils as eutils
from models import utils as mutils
from dynamics import get_vpsde, get_joint_vf, get_joint_stoch_vf, get_avg_vf
from models import ddpm


def init_model(key, config, workdir):
  key, init_key = random.split(key)
  model, initial_params = mutils.init_model(init_key, config)
  optimizer = tutils.get_optimizer(config)
  opt_state = optimizer.init(initial_params)
  state = mutils.State(step=1, opt_state=opt_state,
                       model_params=initial_params,
                       ema_rate=config.model.ema_rate,
                       params_ema=initial_params,
                       sampler_state=0.5,
                       key=key, wandbid=np.random.randint(int(1e7),int(1e8)))

  checkpoint_dir = os.path.join(workdir, "checkpoints")
  mgr_options = orbax.checkpoint.CheckpointManagerOptions(
    create=True, max_to_keep=50, step_prefix='chkpt')
  ckpt_mgr = orbax.checkpoint.CheckpointManager(
    checkpoint_dir, orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler()), mgr_options)
  ckpt_mgr.reload()
  # read preemptied run
  if ckpt_mgr.latest_step() is not None:
    restore_args = flax.training.orbax_utils.restore_args_from_target(state, mesh=None)
    state = ckpt_mgr.restore(ckpt_mgr.latest_step(), items=state, restore_kwargs={'restore_args': restore_args})
  return state, ckpt_mgr, optimizer, model

def train(config, workdir):
  key = random.PRNGKey(config.seed)

  # init model
  state, ckpt_mgr, optimizer, model = init_model(key, config, workdir)
  initial_step = int(state.step)
  key = state.key

  if jax.process_index() == 0:
    wandb.login()
    wandb.init(id=str(state.wandbid), 
               project=config.data.task + '_' + config.data.dataset, 
               resume="allow",
               config=json.loads(config.to_json_best_effort()))
    os.environ["WANDB_RESUME"] = "allow"
    os.environ["WANDB_RUN_ID"] = str(state.wandbid)
  
  # init functions
  q_t, loss_fn, vector_field = get_vpsde(config, model, train=True)
  step_fn = tutils.get_step_fn(optimizer, loss_fn)
  step_fn = jax.pmap(step_fn, axis_name='batch')
  artifact_generator = eutils.get_generator([model], config, jax.jit(vector_field), train=True)
  artifact_generator = jax.vmap(artifact_generator, axis_name='batch')
  # bpd_estimator = eutils.get_bpd_estimator(model, config, vector_field, use_ema=False)
  # bpd_estimator = jax.pmap(bpd_estimator, axis_name='batch')

  # init dataloaders
  train_ds, _, _ = datasets.get_dataset(config,
                                        uniform_dequantization=config.data.uniform_dequantization)
  train_iter = iter(train_ds)
  scaler = datasets.get_image_scaler(config)
  inverse_scaler = datasets.get_image_inverse_scaler(config)

  # run train
  assert (config.train.n_iters % config.train.save_every) == 0

  pstate = flax_utils.replicate(state)
  key = jax.random.fold_in(key, jax.process_index())
  for step in range(initial_step, config.train.n_iters+1):
    batch = jax.tree_map(lambda x: x._numpy(), next(train_iter))
    batch['image'] = scaler(batch['image'])
    key, *next_key = random.split(key, num=jax.local_device_count() + 1)
    next_key = jnp.asarray(next_key)
    (_, pstate), ploss = step_fn((next_key, pstate), batch)
    loss = ploss.mean()

    if (step % config.train.log_every == 0) and (jax.process_index() == 0):
      wandb.log(dict(loss=loss.item()), step=step)

    if (step % config.train.save_every == 0) and (jax.process_index() == 0):
      saved_state = flax_utils.unreplicate(pstate)
      saved_state = saved_state.replace(key=key)
      save_args = flax.training.orbax_utils.save_args_from_target(saved_state)
      ckpt_mgr.save(step // config.train.save_every, saved_state, save_kwargs={'save_args': save_args})

    if step % config.train.eval_every == 0:
      key, *next_keys = random.split(key, num=jax.local_device_count() + 1)
      next_keys = jnp.asarray(next_keys)
      labels = jnp.tile(jnp.arange(10), 10).reshape(jax.local_device_count(),-1)
      artifacts, num_steps = artifact_generator(next_keys, labels, pstate)
      artifacts = artifacts.reshape(-1,
                                    config.data.image_size,
                                    config.data.image_size,
                                    config.data.num_channels)[:config.eval.artifact_size]
      artifacts = inverse_scaler(artifacts)
      
      # key, *next_keys = random.split(key, num=jax.local_device_count() + 1)
      # next_keys = jnp.asarray(next_keys)
      # bpd, num_steps = bpd_estimator(next_keys, pstate, batch)
      wandb.log(dict(examples=[wandb.Image(tutils.stack_imgs(artifacts))],
                     nfe=jnp.mean(num_steps).item()), step=step)
                    #  bpd=jnp.mean(bpd).item()), step=step)


def evaluate_fid(config, workdir, eval_folder, stoch):
  key = random.PRNGKey(config.seed)

  # init model
  state, ckpt_mgr, optimizer, model = init_model(key, config, workdir)
  
  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)
  if stoch:
    sample_dir = 'samples_stoch'
  else:
    sample_dir = 'samples'
  sample_dir = os.path.join(eval_dir, sample_dir)
  tf.io.gfile.makedirs(sample_dir)

  # init generator
  vector_field = get_avg_vf(key, [model], [state], stoch=stoch)
  artifact_generator = eutils.get_generator([model], config, jax.jit(vector_field))
  artifact_generator = jax.vmap(artifact_generator, axis_name='batch')
  inverse_scaler = datasets.get_image_inverse_scaler(config)
  artifact_shape = [config.eval.batch_size, 
                    config.data.image_size, 
                    config.data.image_size, 
                    config.data.num_channels]

  # init inception
  inception_model = evaluation.get_inception_model()

  # generate samples
  num_batches = math.ceil(config.eval.num_samples / config.eval.batch_size)
  for batch_id in range(num_batches):
    key, *next_keys = random.split(key, num=jax.local_device_count() + 1)
    next_keys = jnp.asarray(next_keys)
    labels = jnp.tile(jnp.arange(10), 10).reshape(jax.local_device_count(),-1)
    artifacts, num_steps = artifact_generator(next_keys, labels)
    artifacts = artifacts.reshape(artifact_shape)
    artifacts = inverse_scaler(artifacts)
    artifacts = jnp.clip(artifacts*255.0, 0.0, 255.0).astype(np.uint8)
    print(f'{batch_id}/{num_batches}, artifacts.shape: {artifacts.shape}, num_steps: {num_steps}', flush=True)
    with tf.io.gfile.GFile(os.path.join(sample_dir, f"samples_{batch_id}.npz"), "wb") as fout:
      io_buffer = io.BytesIO()
      np.savez_compressed(io_buffer, samples=artifacts, num_steps=num_steps)
      fout.write(io_buffer.getvalue())

    gc.collect()
    latents = evaluation.run_inception_distributed(artifacts, inception_model)
    gc.collect()
    with tf.io.gfile.GFile(os.path.join(sample_dir, f"statistics_{batch_id}.npz"), "wb") as fout:
      io_buffer = io.BytesIO()
      np.savez_compressed(io_buffer, pool_3=latents)
      fout.write(io_buffer.getvalue())
  
  all_pools = []
  stats = tf.io.gfile.glob(os.path.join(sample_dir, "statistics_*.npz"))
  for stat_file in stats:
    with tf.io.gfile.GFile(stat_file, "rb") as fin:
      stat = np.load(fin)
      all_pools.append(stat["pool_3"])
  all_pools = np.concatenate(all_pools, axis=0)[:config.eval.num_samples]

  train_pools = evaluation.load_dataset_stats(config, eval=False)
  train_fid = evaluation.fid(train_pools["pool_3"], all_pools)
  test_pools = evaluation.load_dataset_stats(config, eval=True)
  test_fid = evaluation.fid(test_pools["pool_3"], all_pools)
  print(f'train FID: {train_fid}, test FID: {test_fid}', flush=True)
  
  with tf.io.gfile.GFile(os.path.join(eval_dir, f"report.npz"), "wb") as f:
    io_buffer = io.BytesIO()
    np.savez_compressed(io_buffer, train_fid=train_fid, test_fid=test_fid)
    f.write(io_buffer.getvalue())


def evaluate_joint_fid(config, workdir, eval_folder, checkpoints, stoch):
  key = random.PRNGKey(config.seed)

  # init model
  states = []
  models = []
  for chkpt in checkpoints:
    state, _, _, model = init_model(key, config, chkpt)
    states.append(state)
    models.append(model)
  
  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)
  if stoch:
    sample_dir = 'samples_stoch'
  else:
    sample_dir = 'samples'
  sample_dir = os.path.join(eval_dir, sample_dir)
  tf.io.gfile.makedirs(sample_dir)

  # init generator
  if stoch:
    vector_field = jax.jit(get_joint_stoch_vf(key, models, states))
  else:
    vector_field = jax.jit(get_joint_vf(key, models, states))
  artifact_generator = eutils.get_generator(models, config, vector_field)
  artifact_generator = jax.vmap(artifact_generator, axis_name='batch')
  inverse_scaler = datasets.get_image_inverse_scaler(config)
  artifact_shape = [config.eval.batch_size, 
                    config.data.image_size, 
                    config.data.image_size, 
                    config.data.num_channels]

  # init inception
  inception_model = evaluation.get_inception_model()

  # generate samples
  num_batches = math.ceil(config.eval.num_samples / config.eval.batch_size)
  for batch_id in range(num_batches):
    key, *next_keys = random.split(key, num=jax.local_device_count() + 1)
    next_keys = jnp.asarray(next_keys)
    labels = jnp.tile(jnp.arange(10), 10).reshape(jax.local_device_count(), -1)
    artifacts, num_steps = artifact_generator(next_keys, labels)
    artifacts = artifacts.reshape(artifact_shape)
    artifacts = inverse_scaler(artifacts)
    artifacts = jnp.clip(artifacts*255.0, 0.0, 255.0).astype(np.uint8)
    print(f'{batch_id}/{num_batches}, artifacts.shape: {artifacts.shape}, num_steps: {num_steps}', flush=True)
    with tf.io.gfile.GFile(os.path.join(sample_dir, f"samples_{batch_id}.npz"), "wb") as fout:
      io_buffer = io.BytesIO()
      np.savez_compressed(io_buffer, samples=artifacts, num_steps=num_steps)
      fout.write(io_buffer.getvalue())

    gc.collect()
    latents = evaluation.run_inception_distributed(artifacts, inception_model)
    gc.collect()
    with tf.io.gfile.GFile(os.path.join(sample_dir, f"statistics_{batch_id}.npz"), "wb") as fout:
      io_buffer = io.BytesIO()
      np.savez_compressed(io_buffer, pool_3=latents)
      fout.write(io_buffer.getvalue())
  
  all_pools = []
  stats = tf.io.gfile.glob(os.path.join(sample_dir, "statistics_*.npz"))
  for stat_file in stats:
    with tf.io.gfile.GFile(stat_file, "rb") as fin:
      stat = np.load(fin)
      all_pools.append(stat["pool_3"])
  all_pools = np.concatenate(all_pools, axis=0)[:config.eval.num_samples]

  train_pools = evaluation.load_dataset_stats(config, eval=False)
  train_fid = evaluation.fid(train_pools["pool_3"], all_pools)
  test_pools = evaluation.load_dataset_stats(config, eval=True)
  test_fid = evaluation.fid(test_pools["pool_3"], all_pools)
  print(f'train FID: {train_fid}, test FID: {test_fid}', flush=True)
  
  with tf.io.gfile.GFile(os.path.join(eval_dir, f"report.npz"), "wb") as f:
    io_buffer = io.BytesIO()
    np.savez_compressed(io_buffer, train_fid=train_fid, test_fid=test_fid)
    f.write(io_buffer.getvalue())


def fid_stats(config, workdir, fid_folder="assets/stats"):
  fid_dir = os.path.join(workdir, fid_folder)
  tf.io.gfile.makedirs(fid_dir)
  
  inception_model = evaluation.get_inception_model()
  
  def get_pools(data_iter):
    all_pools = []
    batch_id = 0
    while True:
      try:
        batch = next(data_iter)
      except StopIteration:
        break
      print("Making FID stats -- step: %d" % (batch_id))
      batch = jax.tree_map(lambda x: x._numpy(), batch)
      batch = (batch['image']*255).astype(np.uint8).reshape((-1, config.data.image_size, config.data.image_size, 3))
      # Force garbage collection before calling TensorFlow code for Inception network
      gc.collect()
      latents = evaluation.run_inception_distributed(batch, inception_model)
      all_pools.append(latents)
      # Force garbage collection again before returning to JAX code
      gc.collect()
      batch_id += 1
    return np.concatenate(all_pools, axis=0)
  
  train_ds, test_ds, dataset_builder = datasets.get_dataset(config,
    additional_dim=None, uniform_dequantization=False, evaluation=True)
  train_iter, test_iter = iter(train_ds), iter(test_ds)

  train_pools = get_pools(train_iter)
  filename = f'{config.data.dataset.lower()}_train_stats.npz'
  with tf.io.gfile.GFile(os.path.join(fid_dir, filename), "wb") as fout:
    io_buffer = io.BytesIO()
    np.savez_compressed(io_buffer, pool_3=train_pools)
    fout.write(io_buffer.getvalue())
  del train_pools
    
  test_pools = get_pools(test_iter)
  filename = f'{config.data.dataset.lower()}_test_stats.npz'
  with tf.io.gfile.GFile(os.path.join(fid_dir, filename), "wb") as fout:
    io_buffer = io.BytesIO()
    np.savez_compressed(io_buffer, pool_3=test_pools)
    fout.write(io_buffer.getvalue())
