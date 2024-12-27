import math

import jax
import jax.random as random
import jax.numpy as jnp

from models import utils as mutils

def sample_time(bs, u0, t_0, t_1):
    u = (u0 + math.sqrt(2)*jnp.arange(bs*jax.device_count())) % 1
    u0=u[-1]
    t = (t_1-t_0)*u[jax.process_index()*bs:(jax.process_index()+1)*bs] + t_0
    return t, u0

def get_vpsde(config, model, train):
  t_0, t_1 = config.data.t_0, config.data.t_1
  
  beta_0 = 0.1
  beta_1 = 20.0
  log_alpha = lambda t: -0.5*t*beta_0-0.25*t**2*(beta_1-beta_0)
  # log_sigma = lambda t: jnp.log(jnp.sqrt(-jnp.expm1(-t*beta_0-0.5*t**2*(beta_1-beta_0))))
  log_sigma = lambda t: jnp.log(t)
  dlog_alphadt = jax.grad(lambda t: log_alpha(t).sum())
  dlog_sigmadt = jax.grad(lambda t: log_sigma(t).sum())
  # beta_t = s_t d/dt log(s_t/alpha_t)
  # beta = lambda t: jnp.exp(log_sigma(t))*(dlog_sigmadt(t) - dlog_alphadt(t))
  beta = lambda t: (1 + 0.5*t*beta_0 + 0.5*t**2*(beta_1-beta_0))
  
  def q_t(key, data, t):
    eps = random.normal(key, shape=data.shape)
    x_t = jnp.exp(log_alpha(t))*data + jnp.exp(log_sigma(t))*eps
    return eps, x_t
  
  def loss(key, params, sampler_state, batch):
    keys = random.split(key, num=2)
    sdlogqdx = mutils.get_model_fn(model, params, train=train)
    data = batch['image']
    labels = batch['label']
    bs = data.shape[0]
    t, next_sampler_state = sample_time(bs, sampler_state, t_0, t_1)
    t = jnp.expand_dims(t, (1,2,3))
    eps, x_t = q_t(keys[0], data, t)
    loss = ((eps + sdlogqdx(t, x_t, labels, keys[1]))**2).sum((1,2,3))
    print(loss.shape, 'final.shape', flush=True)
    return loss.mean(), next_sampler_state
  
  # v_t(x) = dlog(alpha)/dt x - s^2_t d/dt log(s_t/alpha_t) dlog q_t(x)/dx
  def vector_field(t, data, args):
    _, labels, dt, state = args['key'], args['labels'], args['dt'], args['state']
    x, _ = data
    sdlogqdx_fn = mutils.get_model_fn(model, state.model_params, train=False)
    sdlogqdx = sdlogqdx_fn(t*jnp.ones((x.shape[0],1,1,1)), x, labels)
    dx = -dt*(dlog_alphadt(t)*x - beta(t)*sdlogqdx)
    return (dx, jnp.zeros((dx.shape[0], 1)))
  
  return q_t, loss, vector_field


def get_joint_vf(key, models, states):
  beta_0 = 0.1
  beta_1 = 20.0
  log_alpha = lambda t: -0.5*t*beta_0-0.25*t**2*(beta_1-beta_0)
  # log_sigma = lambda t: jnp.log(jnp.sqrt(-jnp.expm1(-t*beta_0-0.5*t**2*(beta_1-beta_0))))
  log_sigma = lambda t: jnp.log(t)
  dlog_alphadt = jax.grad(lambda t: log_alpha(t).sum())
  dlog_sigmadt = jax.grad(lambda t: log_sigma(t).sum())
  # beta_t = s_t d/dt log(s_t/alpha_t)
  # beta = lambda t: jnp.exp(log_sigma(t))*(dlog_sigmadt(t) - dlog_alphadt(t))
  beta = lambda t: (1 + 0.5*t*beta_0 + 0.5*t**2*(beta_1-beta_0))
  nets = []
  for i in range(len(models)):
    nets.append(mutils.get_model_fn(models[i], states[i].params_ema, train=False))

  def joint_vf(t, data, args):
    key, labels, dt = args['key'], args['labels'], args['dt']
    key = jax.random.fold_in(key, t*10_000)
    x, logq = data
    vfs = []
    dlogdx = []
    div = []
    for i in range(len(nets)):
      key, iter_key = jax.random.split(key)
      eps = jax.random.randint(iter_key, x.shape, 0, 2).astype(float)*2 - 1.0
      sdlogdx_fn = lambda _x: nets[i](t*jnp.ones((_x.shape[0],1,1,1)), _x, labels)
      sdlogdx, jvp_val = jax.jvp(sdlogdx_fn, (x,), (eps,))
      vfs.append(dlog_alphadt(t)*x - beta(t)*sdlogdx)
      dlogdx.append(sdlogdx/(t+1e-3))
      div.append(-beta(t)*(jvp_val*eps).sum((1,2,3)))
    vfs, dlogdx, div = jnp.stack(vfs), jnp.stack(dlogdx), jnp.stack(div)
    weights = jax.nn.softmax(1e6*logq)
    dx = -dt*(jnp.expand_dims(weights.T, (2,3,4))*vfs).sum(0)
    dlogq = -(-dt)*div + (dlogdx*(dx[None, ...] - (-dt)*vfs)).sum((2,3,4))
    dlogq = dlogq.T
    dlogq -= jnp.max(dlogq, axis=1, keepdims=True)
    return (dx, dlogq)

  return joint_vf


def get_joint_stoch_vf(key, models, states):
  beta_0 = 0.1
  beta_1 = 20.0
  log_alpha = lambda t: -0.5*t*beta_0-0.25*t**2*(beta_1-beta_0)
  # log_sigma = lambda t: jnp.log(jnp.sqrt(-jnp.expm1(-t*beta_0-0.5*t**2*(beta_1-beta_0))))
  log_sigma = lambda t: jnp.log(t)
  dlog_alphadt = jax.grad(lambda t: log_alpha(t).sum())
  dlog_sigmadt = jax.grad(lambda t: log_sigma(t).sum())
  # beta_t = s_t d/dt log(s_t/alpha_t)
  # beta = lambda t: jnp.exp(log_sigma(t))*(dlog_sigmadt(t) - dlog_alphadt(t))
  beta = lambda t: (1 + 0.5*t*beta_0 + 0.5*t**2*(beta_1-beta_0))
  nets = []
  for i in range(len(models)):
    nets.append(mutils.get_model_fn(models[i], states[i].params_ema, train=False))

  def joint_vf(t, data, args):
    key, labels, dt = args['key'], args['labels'], args['dt']
    key = jax.random.fold_in(key, t*10_000)
    x, logq = data
    sscores = []
    for i in range(len(nets)):
      sdlogdx = nets[i](t*jnp.ones((x.shape[0],1,1,1)), x, labels)
      sscores.append(sdlogdx)
    sscores = jnp.stack(sscores)
    weights = jax.nn.softmax(1e6*logq)
    balanced_sscore = (jnp.expand_dims(weights.T, (2,3,4))*sscores).sum(0)
    eps = random.normal(key, shape=x.shape)
    dx = -dt*(dlog_alphadt(t)*x - 2*beta(t)*balanced_sscore) + jnp.sqrt(2*jnp.exp(log_sigma(t))*beta(t)*dt)*eps
    # dlogq = -(dx[None,...] + dt*(dlog_alphadt(t)*x[None,...] - 2*beta(t)*sscores))**2
    # dlogq += (dx[None,...] + dt*dlog_alphadt(t)*(x+dx)[None,...])**2
    # dlogq /= 4*jnp.exp(log_sigma(t))*beta(t)*dt
    dlogq = (dlog_alphadt(t)*(x+dx)[None,...] - (dlog_alphadt(t)*x[None,...] - 2*beta(t)*sscores))
    dlogq *= (dt*(dlog_alphadt(t)*x[None,...] - 2*beta(t)*sscores) + 2*dx[None,...] + dt*dlog_alphadt(t)*(x+dx)[None,...])
    dlogq /= 4*jnp.exp(log_sigma(t))*beta(t)
    dlogq = dlogq.sum((2,3,4)).T
    dlogq -= jnp.max(dlogq, axis=1, keepdims=True)
    return (dx, dlogq)
  return joint_vf


def get_avg_vf(key, models, states, stoch=True):
  beta_0 = 0.1
  beta_1 = 20.0
  log_alpha = lambda t: -0.5*t*beta_0-0.25*t**2*(beta_1-beta_0)
  # log_sigma = lambda t: jnp.log(jnp.sqrt(-jnp.expm1(-t*beta_0-0.5*t**2*(beta_1-beta_0))))
  log_sigma = lambda t: jnp.log(t)
  dlog_alphadt = jax.grad(lambda t: log_alpha(t).sum())
  dlog_sigmadt = jax.grad(lambda t: log_sigma(t).sum())
  # beta_t = s_t d/dt log(s_t/alpha_t)
  # beta = lambda t: jnp.exp(log_sigma(t))*(dlog_sigmadt(t) - dlog_alphadt(t))
  beta = lambda t: (1 + 0.5*t*beta_0 + 0.5*t**2*(beta_1-beta_0))
  nets = []
  for i in range(len(models)):
    nets.append(mutils.get_model_fn(models[i], states[i].params_ema, train=False))

  def joint_vf(t, data, args):
    key, labels, dt = args['key'], args['labels'], args['dt']
    key = jax.random.fold_in(key, t*10_000)
    x, _ = data
    vfs = []
    for i in range(len(nets)):
      sdlogdx = nets[i](t*jnp.ones((x.shape[0],1,1,1)), x, labels)
      if stoch:
        vfs.append(dlog_alphadt(t)*x - 2*beta(t)*sdlogdx)
      else:
        vfs.append(dlog_alphadt(t)*x - beta(t)*sdlogdx)
    vfs = jnp.stack(vfs)
    dx = -dt*vfs.mean(0) 
    if stoch:
      eps = random.normal(key, shape=dx.shape)
      dx += jnp.sqrt(2*jnp.exp(log_sigma(t))*beta(t)*dt)*eps
    return (dx, jnp.zeros((dx.shape[0], len(models))))

  return joint_vf

# def get_joint_vf(key, models, states):
#   beta_0 = 0.1
#   beta_1 = 20.0
#   log_alpha = lambda t: -0.5*t*beta_0-0.25*t**2*(beta_1-beta_0)
#   # log_sigma = lambda t: jnp.log(jnp.sqrt(-jnp.expm1(-t*beta_0-0.5*t**2*(beta_1-beta_0))))
#   log_sigma = lambda t: jnp.log(t)
#   dlog_alphadt = jax.grad(lambda t: log_alpha(t).sum())
#   dlog_sigmadt = jax.grad(lambda t: log_sigma(t).sum())
#   # beta_t = s_t d/dt log(s_t/alpha_t)
#   # beta = lambda t: jnp.exp(log_sigma(t))*(dlog_sigmadt(t) - dlog_alphadt(t))
#   beta = lambda t: (1 + 0.5*t*beta_0 + 0.5*t**2*(beta_1-beta_0))
#   nets = []
#   for i in range(len(models)):
#     nets.append(mutils.get_model_fn(models[i], states[i].params_ema, train=False))

#   def joint_vf(t, data, args):
#     key, labels = args['key'], args['labels']
#     key = jax.random.fold_in(key, t*10_000)
#     x, logq = data
#     vfs = []
#     dlogdx = []
#     div = []
#     for i in range(len(nets)):
#       key, iter_key = jax.random.split(key)
#       eps = jax.random.randint(iter_key, x.shape, 0, 2).astype(float)*2 - 1.0
#       sdlogdx_fn = lambda _x: nets[i](t*jnp.ones((_x.shape[0],1,1,1)), _x, labels)
#       sdlogdx, jvp_val = jax.jvp(sdlogdx_fn, (x,), (eps,))
#       vfs.append(dlog_alphadt(t)*x - beta(t)*sdlogdx)
#       dlogdx.append(sdlogdx/(t+1e-3))
#       div.append(-beta(t)*(jvp_val*eps).sum((1,2,3)))
#     vfs, dlogdx, div = jnp.stack(vfs), jnp.stack(dlogdx), jnp.stack(div)
#     weights = jax.nn.softmax(logq)
#     dxdt = (jnp.expand_dims(weights.T, (2,3,4))*vfs).sum(0)
#     dlogqdt = -div + (dlogdx*(dxdt[None, ...] - vfs)).sum((2,3,4))
#     dlogqdt = dlogqdt.T
#     dlogqdt -= jnp.max(dlogqdt, axis=1, keepdims=True)
#     return (dxdt, dlogqdt)

#   return joint_vf


# def get_avg_vf(key, models, states):
#   beta_0 = 0.1
#   beta_1 = 20.0
#   log_alpha = lambda t: -0.5*t*beta_0-0.25*t**2*(beta_1-beta_0)
#   # log_sigma = lambda t: jnp.log(jnp.sqrt(-jnp.expm1(-t*beta_0-0.5*t**2*(beta_1-beta_0))))
#   log_sigma = lambda t: jnp.log(t)
#   dlog_alphadt = jax.grad(lambda t: log_alpha(t).sum())
#   dlog_sigmadt = jax.grad(lambda t: log_sigma(t).sum())
#   # beta_t = s_t d/dt log(s_t/alpha_t)
#   # beta = lambda t: jnp.exp(log_sigma(t))*(dlog_sigmadt(t) - dlog_alphadt(t))
#   beta = lambda t: (1 + 0.5*t*beta_0 + 0.5*t**2*(beta_1-beta_0))
#   nets = []
#   for i in range(len(models)):
#     nets.append(mutils.get_model_fn(models[i], states[i].params_ema, train=False))

#     def joint_vf(t, data, args):
#       _, labels = args['key'], args['labels']
#       x, _ = data
#       vfs = []
#       for i in range(len(nets)):
#         sdlogdx_fn = lambda _x: nets[i](t*jnp.ones((_x.shape[0],1,1,1)), _x, labels)
#         vfs.append(dlog_alphadt(t)*x - beta(t)*sdlogdx_fn(x))
#       vfs = jnp.stack(vfs)
#       dxdt = vfs.mean(0)
#       return (dxdt, 0.0)

#   return joint_vf

