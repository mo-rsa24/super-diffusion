from functools import partial
import math

import jax
import jax.numpy as jnp
import flax
import optax
import diffrax
import numpy as np

from models import utils as mutils

def get_optimizer(config):
  schedule = optax.join_schedules([optax.linear_schedule(0.0, config.train.lr, config.train.warmup), 
                                   optax.constant_schedule(config.train.lr)], 
                                   boundaries=[config.train.warmup])
  optimizer = optax.adam(learning_rate=schedule, b1=config.train.beta1, eps=config.train.eps)
  optimizer = optax.chain(
    optax.clip(config.train.grad_clip),
    optimizer
  )
  return optimizer


def get_step_fn(optimizer, loss_fn):

  def step_fn(carry_state, batch):
    (key, state) = carry_state
    key, iter_key = jax.random.split(key)
    grad_fn = jax.value_and_grad(loss_fn, argnums=1, has_aux=True)
    (loss, new_sampler_state), grad = grad_fn(iter_key, state.model_params, state.sampler_state, batch)
    grad = jax.lax.pmean(grad, axis_name='batch')
    updates, opt_state = optimizer.update(grad, state.opt_state, state.model_params)
    new_params = optax.apply_updates(state.model_params, updates)
    new_params_ema = jax.tree_map(
      lambda p_ema, p: p_ema * state.ema_rate + p * (1. - state.ema_rate),
      state.params_ema, new_params
    )
    new_state = state.replace(
      step=state.step+1,
      opt_state=opt_state,
      sampler_state=new_sampler_state, 
      model_params=new_params,
      params_ema=new_params_ema
    )

    loss = jax.lax.pmean(loss, axis_name='batch')
    new_carry_state = (key, new_state)
    return new_carry_state, loss

  return step_fn


def stack_imgs(x, n=8, m=8):
    im_size = x.shape[2]
    big_img = np.zeros((n*im_size,m*im_size,x.shape[-1]),dtype=np.uint8)
    for i in range(n):
        for j in range(m):
            p = x[i*m+j] * 255
            p = p.clip(0, 255).astype(np.uint8)
            big_img[i*im_size:(i+1)*im_size, j*im_size:(j+1)*im_size, :] = p
    return big_img
