import numpy as np
import torch
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    quaternion_to_matrix,
    quaternion_to_axis_angle,
    matrix_to_quaternion,
)
import GPUtil

# Proteus imports 
from proteus_data import so3_diffuser as proteus_so3_diffuser
from proteus_data import r3_diffuser as proteus_r3_diffuser
from proteus_data import se3_diffuser as proteus_se3_diffuser
from proteus_data import all_atom, protein

from se3diff_data import so3_diffuser as framediff_so3_diffuser
from se3diff_data import r3_diffuser as framediff_r3_diffuser
from se3diff_data import se3_diffuser as framediff_se3_diffuser

from proteus_data import utils as du
from proteus_data.se3_diffuser import _extract_trans_rots, _assemble_rigid, SE3Diffuser
from proteus_openfold.utils import rigid_utils as ru

from scipy.spatial.transform import Rotation
from einops import rearrange, repeat
from tqdm import tqdm


import logging, copy, tree, os
from functorch import grad
import collections

import wandb


class ScoreWrapper(torch.nn.Module):
    def __init__(self, model, model_name, diffuser):
        super(ScoreWrapper, self).__init__()
        self.model = model[model_name]["model"]
        self.model_name = model_name
        self.diffuser = diffuser

    def forward(self, x, *args, **kwargs):
        input_feats = args[0]
        component = args[1]

        if component == "trans":
            input_feats["rigids_t"] = torch.cat(
                (input_feats["rigids_t"][:, :, :4], x), dim=-1
            )
        elif component == "rots":
            input_feats["rigids_t"] = torch.cat(
                (x, input_feats["rigids_t"][:, :, 4:]), dim=-1
            )

        if self.model_name == "proteus":
            self_condition = args[2]
            struct2seq = args[3]
            model_out = self.model(
                input_feats, self_condition=self_condition, struct2seq=struct2seq
            )
        else:
            model_out = self.model(input_feats)

        out = model_out["rigids"]
        if self.model_name == "proteus":
            out = out.to_tensor_7()
        if component == "trans":
            if self.model_name == "proteus":
                score = self.diffuser.calc_trans_score(
                    ru.Rigid.from_tensor_7(input_feats["rigids_t"]).get_trans(),
                    model_out["pred_trans"],
                    input_feats["t"],
                    use_torch=True,
                    scale=True,
                )
            else:
                score = model_out["trans_score"]
            return score

        elif component == "rots":
            if self.model_name == "proteus":
                score = self.diffuser.calc_rot_score(
                    ru.Rigid.from_tensor_7(input_feats["rigids_t"]).get_rots(),
                    ru.Rotation(model_out["pred_rotmats"]),
                    input_feats["t"],
                )
            else:
                score = model_out["rot_score"]

            # [1, 100, 3] .... [1, 100, 3, 3] we can get this too
            score_rotvec = score
            score = axis_angle_to_matrix(score_rotvec)
            return score


class CompositionDiffusion:
    """_summary_: 
    - TODO: Write a summary
    """
    def __init__(
        self, 
        comp_diff_conf=None,
        models=None,
        ):
        
        self.comp_diff_conf = comp_diff_conf

        self.logger = logging.getLogger(__name__)

        if torch.cuda.is_available():
            if self.comp_diff_conf.inference.gpu_id is None:
                available_gpus = "".join(
                    [str(x) for x in GPUtil.getAvailable(order="memory", limit=8)]
                )
                self.device = f"cuda:{available_gpus[0]}"
            else:
                self.device = f"cuda:{self.comp_diff_conf.inference.gpu_id}"
        else:
            self.device = "cpu"

        # initialize models
        self.models = models


        self.stochastic = self.comp_diff_conf.inference.stochastic
        
        self.framediff_forward_model = self.models["framediff"]["model"]
        self.init_feats_framediff = self.models["framediff"]["init_feats"]
        self.framediff_conf = self.models['framediff']['conf']
        
        self.proteus_conf = self.models['proteus']['conf']
        self.proteus_forward_model = self.models['proteus']['model']
        self.init_feats_proteus = self.models["proteus"]["init_feats"]
        
        # initialize SO3 and R3 diffusers
        self._diffuse_rot = comp_diff_conf.diffuser.diffuse_rot
        self.so3_diffuser = None 

        self._diffuse_trans = comp_diff_conf.diffuser.diffuse_trans
        self.r3_diffuser = None 
        
        
        self.diffuser = SE3Diffuser(self.comp_diff_conf.diffuser)
        self.mixing_method = self.comp_diff_conf.inference.mixing_method
        assert self.mixing_method in ["composition", "mixture", "baseline_proteus", "baseline_framediff"]
        self.proteus_r3_diffuser = proteus_r3_diffuser.R3Diffuser(self.comp_diff_conf.diffuser.r3, stochastic=self.stochastic)
        self.proteus_so3_diffuser = proteus_so3_diffuser.SO3Diffuser(self.comp_diff_conf.diffuser.so3, stochastic=self.stochastic)
        self.proteus_se3_diffuser  = proteus_se3_diffuser.SE3Diffuser(self.proteus_conf.diffuser, stochastic=self.stochastic)
        self.framediff_r3_diffuser = framediff_r3_diffuser.R3Diffuser(self.comp_diff_conf.diffuser.r3, stochastic=self.stochastic)
        self.framediff_so3_diffuser = framediff_so3_diffuser.SO3Diffuser(self.comp_diff_conf.diffuser.so3, stochastic=self.stochastic)
        self.framediff_se3_diffuser  = framediff_se3_diffuser.SE3Diffuser(self.framediff_conf.diffuser, stochastic=self.stochastic)

        self.r3_diffuser = self.proteus_r3_diffuser
        self.so3_diffuser = self.proteus_so3_diffuser
        self.se3_diffuser = self.framediff_se3_diffuser

        self.framediff_model = ScoreWrapper(self.models, "framediff", self.se3_diffuser)
        self.proteus_model = ScoreWrapper(self.models, "proteus", self.se3_diffuser)

        
        self.alpha_trans = 0.5 * torch.ones(self.comp_diff_conf.data.num_t + 1, device=self.device)
        self.alpha_rots = 0.5 * torch.ones(self.comp_diff_conf.data.num_t + 1, device=self.device)
        self.kappa = self.comp_diff_conf.inference.kappa
        self.eta = 0.0  # 0.9

        self.noise_scale = self.comp_diff_conf.experiment.noise_scale
        
        self.num_inference_steps = self.comp_diff_conf.data.num_t # + 1
        self.min_t = self.comp_diff_conf.data.min_t
        self.inference_steps = np.linspace(
            self.min_t, 1.0, self.num_inference_steps
        )[::-1]
        
        # initialize arrays for log-likelihoods
        self.ll_proteus_trans = torch.zeros((1, self.num_inference_steps + 1)).to(self.device)
        self.ll_proteus_rots = torch.zeros((1, self.num_inference_steps + 1)).to(self.device)
        self.ll_framediff_trans = torch.zeros((1, self.num_inference_steps + 1)).to(self.device)
        self.ll_framediff_rots = torch.zeros((1, self.num_inference_steps + 1)).to(self.device)

        self.val_dict = {}
        self.esm_rate = self.proteus_conf.inference.diffusion.rate_t_esm_condition
        self.num_t_esm = int(self.esm_rate * self.num_inference_steps)
        self.reverse_steps = np.linspace(self.min_t, 1.0, self.num_inference_steps)[::-1]
        self.reverse_steps_esm = self.reverse_steps[np.linspace(0, self.num_inference_steps-1, self.num_t_esm, dtype=int)]
        self.dt = 1/self.num_inference_steps

        self.T_trans = self.comp_diff_conf.inference.temp_trans
        self.T_rots = self.comp_diff_conf.inference.temp_rots
        self.logp_trans = self.comp_diff_conf.inference.logp_trans
        self.logp_rots = self.comp_diff_conf.inference.logp_rots
        self.kappa_operator = self.comp_diff_conf.inference.kappa_operator
        assert self.kappa_operator in ["OR", "AND"]
    
    def reset_val_dict(self):
        self.val_dict = {'proteus': {'rots': {}, 'trans': {},},
                         'framediff': {'rots': {}, 'trans': {},}}
    
    def _apply_mask(self, x_diff, x_fixed, diff_mask):
        return diff_mask * x_diff + (1 - diff_mask) * x_fixed

    def check_assert(self, tensor_list, tgt_shape):
        for tensor in tensor_list:
            assert list(tensor.shape) == tgt_shape, (list(tensor.shape), tgt_shape)

    def one_step_proteus(self, i, t, latents, t_placeholder, diffuse_mask, model_out):
        rot_score, trans_score, rigid_pred = None, None, None

        self_condition = self.comp_diff_conf.interpolant.self_condition
        with torch.no_grad():
            model_out = None if not self_condition else model_out
            if t> self.min_t:
                latents = self._set_t_feats(latents, t, t_placeholder)

            model_out = self.proteus_forward_model(latents,self_condition=model_out,struct2seq=True if t in self.reverse_steps_esm else False)
            
            if t > self.min_t:
                rot_score = self.se3_diffuser.calc_rot_score(
                    ru.Rigid.from_tensor_7(latents['rigids_t']).get_rots(),
                    ru.Rotation(model_out['pred_rotmats']),
                    latents['t']
                )
                trans_score = self.se3_diffuser.calc_trans_score(
                    ru.Rigid.from_tensor_7(latents['rigids_t']).get_trans(),
                    model_out['pred_trans'],
                    latents['t'],
                    use_torch = True,
                )
                rigid_pred = model_out['rigids']

        return  model_out, rot_score, trans_score, rigid_pred
                
    def one_step_framediff(self, i, t, latents, t_placeholder):
        rot_score, trans_score, rigid_pred = None, None, None
        if i == 0 and self.framediff_forward_model._model_conf.embed.embed_self_conditioning:
            latents = self._set_t_feats(
                latents, self.reverse_steps[0], t_placeholder)
            latents = self._self_conditioning(latents, self.framediff_forward_model)
            
        with torch.no_grad():
            if t > self.min_t:
                latents = self._set_t_feats(latents, t, t_placeholder)
             
                model_out = self.framediff_forward_model(latents) 
                rigid_pred = model_out['rigids']
                rot_score = model_out['rot_score']
                trans_score = model_out['trans_score']
                if self.framediff_forward_model._model_conf.embed.embed_self_conditioning:
                    latents['sc_ca_t'] = rigid_pred[..., 4:]
                fixed_mask = latents['fixed_mask'] * latents['res_mask']
                diffuse_mask = (1 - latents['fixed_mask']) * latents['res_mask']
            else:
                model_out = self.framediff_forward_model(latents)

        return  model_out, rot_score, trans_score, rigid_pred
                    
    
    def get_vel(self, latents, model_name, component, get_div=False, model_out=None, struct2seq=None):
        for k in latents.keys():
            latents[k] = latents[k].detach()
        latents_rigids_t_copy = latents['rigids_t'].clone()
        
        if model_name == "framediff":
            with torch.no_grad():
                v = lambda _x: self.framediff_model(
                    ru.rot_to_quat(_x) if component == "rots" else _x,
                    latents,
                    component,
                )
        elif model_name == "proteus":
            assert model_out is not None
            assert struct2seq is not None
            with torch.no_grad():
                v = lambda _x: self.proteus_model(
                    ru.rot_to_quat(_x) if component == "rots" else _x,
                    latents,
                    component,
                    model_out,
                    struct2seq,
                )

        # latents['rigids_t'] is a [1, N, 7]-dim tensor where [:, :, :4] is rot quaternion and [:, :, 4:] is translation
        # code below extracts part of tensor depending on whether we are working for translations or rotations
        if component == "trans":
            component_latent_trans = latents_rigids_t_copy[:, :, 4:]
        elif component == "rots":
            component_latent = latents_rigids_t_copy[:, :, :4]
            comp_latent_rotmats = ru.quat_to_rot(component_latent) # [1, 100, 3, 3]
            u, _, vT = torch.linalg.svd(comp_latent_rotmats)
            prj_comp_latent = u @ vT # [1, 100, 3, 3]

        if get_div:
            with torch.no_grad():
                z = prj_comp_latent if component == "rots" else component_latent_trans
                eps = torch.randint_like(z, 2, dtype=z.dtype) * 2 - 1
                vel, div = torch.autograd.functional.jvp(
                    v,
                    z,
                    eps,
                ) # vel is output of calling v(component_latent); div is dot product of dv(component_latent)/dcomponent_latent and component_latent
                # div is shape [1, 100, 3], eps is shape [1, 100, 3, 3]
                div = -(eps*div).sum((1,2,3)) if len(div.shape) > 3 else -(eps*div).sum((1,2)) # changed from (1,2,3) because output is 3D not 4D
                
        else:
            print("no get_div mode doesn't work yet"); exit()
        vel = vel.chunk(len(z)) 
        div = div.chunk(len(z))
        return vel, div
    
    def compute_vels(self, latents, proteus_model_out, proteus_struct2seq):
        for model_name in self.val_dict:
            for component in self.val_dict[model_name]:
                model_out = None
                struct2seq = None
                
                if model_name == "proteus":
                    model_out = proteus_model_out
                    struct2seq = proteus_struct2seq

                (_,), (divlog,) = self.get_vel(
                    latents,
                    model_name=model_name,
                    component=component,
                    get_div=True,
                    model_out=model_out,
                    struct2seq=struct2seq,
                )
                self.check_assert([divlog], tgt_shape=[1])
                self.val_dict[model_name][component]["divlog"] = divlog
    def compute_stoch_dll(self, t, x, dx, component, f_x):

        for model_name in self.val_dict:
            score = self.val_dict[model_name][component]['score']

            ndim = (score.shape[1] * score.shape[2])

            if component == "trans":
                beta_t = 0.5 * self.r3_diffuser.diffusion_coef(t)**2
                beta_t = torch.tensor(beta_t).to(self.device)

                dlog_alphadt = -1/2 * self.r3_diffuser.b_t(t)
                                   
                output = ndim * self.dt * dlog_alphadt
                output += ((dx + self.dt*(f_x - beta_t*score))*score)
            elif component == "rots":
                beta_t = 0.5 * self.so3_diffuser.diffusion_coef(t)**2
                beta_t = torch.tensor(beta_t).to(self.device)

                output = -self.dt * beta_t*score**2
                output += ((dx) * score)

            output = output.sum()
            self.val_dict[model_name][component]['dlldt'] = output


    def compute_dll(self, t, vel_mixed, dlog_alphadt_x, beta_t, component):
        for model_name in self.val_dict:
            score = self.val_dict[model_name][component]['score']
            div = self.val_dict[model_name][component]['divlog']

            if component == "trans":
                v = dlog_alphadt_x - beta_t * score
                dlldt = (
                    -dlog_alphadt_x * score.shape[-1] * score.shape[-2] + (beta_t) * div
                )
            elif component == "rots":
                v = -beta_t * score
                dlldt = beta_t * div

            dlldt += -((score) * (v - vel_mixed)).sum(
                (1, 2), keepdims=True
            ).squeeze()
            
            self.val_dict[model_name][component]['dlldt'] = dlldt.sum()

    def kappa_OR(self, i, t, component):
        if component == "trans":
            log_like_1 = self.ll_proteus_trans[:, i]
            log_like_2 = self.ll_framediff_trans[:, i]
            T = self.T_trans
            logp=self.logp_trans
        elif component == "rots":
            log_like_1 = self.ll_proteus_rots[:, i]
            log_like_2 = self.ll_framediff_rots[:, i]
            T = self.T_rots
            logp=self.logp_rots
        kappa = torch.softmax(torch.stack([T*(log_like_1+logp), T*(log_like_2)]), 0)[0]
        return kappa

    def kappa_AND(self, i, t, component, beta_t, eps, f_x):
        sdlogdx_1 = self.val_dict['proteus'][component]['score'].to(dtype=torch.float64)
        sdlogdx_2 = self.val_dict['framediff'][component]['score'].to(dtype=torch.float64)

        dim = (sdlogdx_1.shape[1] * sdlogdx_2.shape[2])
        if component == "trans":
            logp=self.logp_trans
            sigma_t = torch.sqrt(self.r3_diffuser.b_t(t))
            max_ = torch.sqrt(torch.tensor(self.comp_diff_conf.diffuser.r3.max_b))
            min_ = torch.sqrt(torch.tensor(self.comp_diff_conf.diffuser.r3.min_b))

        elif component == "rots":
            logp=self.logp_rots
            sigma_t = self.sigma_r(t)
            max_ = torch.tensor(self.comp_diff_conf.diffuser.so3.max_sigma)
            min_ = torch.tensor(self.comp_diff_conf.diffuser.so3.min_sigma)

        sigma_t = -0.5*dim*torch.log(sigma_t)
        min_sigma = -0.5*dim*torch.log(max_)
        max_sigma = -0.5*dim*torch.log(min_)
        sigma_t = (sigma_t - min_sigma) / (max_sigma - min_sigma)

        wandb.log({f"sigma_t_{component}": sigma_t,
                   f"sigma_min_{component}": min_sigma,
                   f"sigma_max_{component}": max_sigma,},
                   step=i)

        noise = torch.sqrt(2*beta_t*self.dt)*eps
        dx_ind = -self.dt*(f_x - 2*beta_t*sdlogdx_2) + noise

        delta_sdlogdx = sdlogdx_1 - sdlogdx_2

        kappa = -self.dt * beta_t * delta_sdlogdx * (sdlogdx_1 + sdlogdx_2)

        kappa += ((dx_ind + self.dt*f_x)*delta_sdlogdx)
        kappa_div = (self.dt*2*beta_t*torch.square(delta_sdlogdx)).sum() 
        kappa /= kappa_div
        kappa = -kappa.sum()

        lift = (logp * sigma_t)/self.num_inference_steps

        kappa += lift / kappa_div
        return kappa
        
    def compute_kappas(self, i, t, x=None, beta_t_trans=None, beta_t_rots=None, eps=None, f_x_trans=None):
        if self.kappa_operator == "OR":
            kappa_trans = self.kappa_OR(i, t, "trans")
            kappa_rots = self.kappa_OR(i, t, "rots")
        elif self.kappa_operator == "AND":
            kappa_trans = self.kappa_AND(i, t, "trans", beta_t_trans, eps, f_x_trans)
            kappa_rots = self.kappa_AND(i, t, "rots", beta_t_rots, eps, 0)
        return kappa_trans, kappa_rots
        
    def latent_mixing(self, latents):
        reverse_steps = np.linspace(self.min_t, 1.0, self.num_inference_steps)[::-1]
        esm_rate = self.proteus_conf.inference.diffusion.rate_t_esm_condition
        num_t_esm = int(esm_rate * self.num_inference_steps)
        reverse_steps_esm = reverse_steps[np.linspace(0, self.num_inference_steps-1, num_t_esm, dtype=int)]

        init_feats = latents

        if latents['rigids_t'].ndim == 2:
            t_placeholder = torch.ones((1,)).to(self.device)
        else:
            t_placeholder = torch.ones(
                (latents['rigids_t'].shape[0],)).to(self.device)

        dt = 1/self.num_inference_steps
        noise_scale = self.noise_scale

        all_msas = []
        all_seqs = []
        all_rigids = [du.move_to_np(copy.deepcopy(latents['rigids_t']))]
        all_rigids_0 = []
        all_bb_prots = []
        all_bb_mask = []
        all_bb_0_pred = []
        
        proteus_model_out, model_out = None, None
        diffuse_mask = (1 - latents['fixed_mask']) * latents['res_mask']
        for i, t in tqdm(enumerate(reverse_steps), colour='MAGENTA'):
            self.reset_val_dict()
            t_ = torch.tensor(t)

            struct2seq =True if t in reverse_steps_esm else False
            proteus_model_out, proteus_rots_score, proteus_trans_score, _ = self.one_step_proteus(i, t, latents,  t_placeholder, diffuse_mask, proteus_model_out)
            _, framediff_rots_score, framediff_trans_score, _ = self.one_step_framediff(i, t, latents, t_placeholder)

            x_t_trans = latents["rigids_t"][:, :, 4:] 
            x_t_trans = self.r3_diffuser._scale(x_t_trans)

            eps=torch.randn(x_t_trans.shape).to(self.device) * self.noise_scale
            if t> self.min_t:
                self.val_dict['proteus']['rots']['score'] = proteus_rots_score
                self.val_dict['proteus']['trans']['score'] = proteus_trans_score
                self.val_dict['framediff']['rots']['score'] = framediff_rots_score
                self.val_dict['framediff']['trans']['score'] = framediff_trans_score
            
            beta_t_trans = 0.5 * self.r3_diffuser.diffusion_coef(t)**2
            beta_t_trans = torch.tensor(beta_t_trans).to(self.device)
            f_x_trans = self.r3_diffuser.drift_coef(x_t_trans, t)

            beta_t_rots = 0.5 * self.so3_diffuser.diffusion_coef(t)**2
            beta_t_rots = torch.tensor(beta_t_rots).to(self.device)


            if self.mixing_method == "baseline_proteus":
                kappa_trans = torch.tensor(1)
                kappa_rots = torch.tensor(1)
            elif self.mixing_method == "baseline_framediff":
                kappa_trans = torch.tensor(0)
                kappa_rots = torch.tensor(0)
            elif self.mixing_method == "mixture":
                self.kappa = float(self.kappa)
                kappa_trans  = torch.tensor(self.kappa)
                kappa_rots = torch.tensor(self.kappa)
            elif self.mixing_method == "composition":
                if t> self.min_t:
                    kappa_trans, kappa_rots = self.compute_kappas(i, t_, x=x_t_trans, beta_t_rots=beta_t_rots, beta_t_trans=beta_t_trans, eps=eps, f_x_trans=f_x_trans)
            
            # compute log-likelihoods
            if t> self.min_t:
                with torch.no_grad():
                   dx_trans = -dt * (f_x_trans
                           - 2*beta_t_trans*(framediff_trans_score + kappa_trans*(proteus_trans_score - framediff_trans_score)))
                   dx_trans +=  torch.sqrt(2*beta_t_trans*dt)*eps
                   #dx_trans = dx_trans.to(dtype=torch.float64)

                   dx_rots = dt*2*beta_t_rots*(framediff_rots_score + kappa_rots*(proteus_rots_score - framediff_rots_score))
                   dx_rots += torch.sqrt(2*beta_t_rots*dt)*eps
                   #dx_rots = dx_rots.to(dtype=torch.float64)
                 
                   if self.mixing_method == "composition":
                       #if diffuse_mask is not None:
                       self.compute_stoch_dll(t, x_t_trans, dx_trans,  "trans", f_x=f_x_trans)
                       self.compute_stoch_dll(t, None, dx_rots, "rots", f_x=0)


                       self.ll_proteus_trans[:, i + 1] = self.ll_proteus_trans[:, i] + self.val_dict['proteus']['trans']['dlldt']
                       self.ll_framediff_trans[:, i + 1] = self.ll_framediff_trans[:, i]  + self.val_dict['framediff']['trans']['dlldt']
                       self.ll_proteus_rots[:, i + 1] = self.ll_proteus_rots[:, i] + self.val_dict['proteus']['rots']['dlldt'] 
                       self.ll_framediff_rots[:, i + 1] = self.ll_framediff_rots[:, i] + self.val_dict['framediff']['rots']['dlldt'] 
                

                   score_trans_mixed = (
                       kappa_trans * proteus_trans_score
                       + (1 - kappa_trans) * framediff_trans_score
                   )
                   score_rots_mixed = (
                       kappa_rots * proteus_rots_score
                       + (1 - kappa_rots) * framediff_rots_score
                   )
                   
                   self.log(i, kappa_trans, kappa_rots)

                # generate trans and tors for next time step
                rigids_t = self.se3_diffuser.reverse(
                    rigid_t=ru.Rigid.from_tensor_7(latents["rigids_t"].detach()),
                    rot_score=du.move_to_np(score_rots_mixed),
                    trans_score=du.move_to_np(score_trans_mixed),
                    t=t,
                    dt=dt,
                    diffuse_mask=du.move_to_np(diffuse_mask),
                    noise_scale=noise_scale,
                    center=True,
                    dx_trans = du.move_to_np(dx_trans),
                    dx_rots = du.move_to_np(dx_rots),
                    )


            
            latents['rigids_t'] = rigids_t.to_tensor_7().to(self.device)
            res_mask = latents['res_mask'][0].detach().cpu()

            all_rigids.append(du.move_to_np(rigids_t.to_tensor_7()))
            
            if self.mixing_method == "baseline":
                all_rigids_0.append(du.move_to_np(rigid_pred.to_tensor_7()))
            else:
                all_rigids_0.append(du.move_to_np(rigids_t.to_tensor_7()))
            
            pred_rots_mats_t = rigids_t.get_rots().get_rot_mats()
            pred_trans_t = rigids_t.get_trans()
            atom37_0 = all_atom.to_atom37(pred_trans_t, pred_rots_mats_t)[0]
            atom37_0 = du.adjust_oxygen_pos(atom37_0, res_mask)
            all_bb_0_pred.append(du.move_to_np(atom37_0))
            all_seqs.append(None)
            atom37_t = all_atom.to_atom37(rigids_t._trans,rigids_t._rots.get_rot_mats())[0]
            atom37_t = du.adjust_oxygen_pos(atom37_t, res_mask)
            all_bb_prots.append(du.move_to_np(atom37_t))
            all_bb_mask.append(du.move_to_np(res_mask))
            all_msas.append(None)
            
        sample_out = {
                "final_atom_positions" : du.move_to_np(atom37_0[None]),
                "final_atom_mask" : du.move_to_np(init_feats["atom37_atom_exists"]),
                "fixed_mask" : du.move_to_np(init_feats["fixed_mask"]),
            }
            
        flip = lambda x: np.flip(np.stack(x), (0,))
        all_msas = flip(all_msas)[:,None]
        all_seqs = flip(all_seqs)[:,None]
        all_bb_prots = flip(all_bb_prots)[:,None]
        all_bb_mask = flip(all_bb_mask)[:,None]
        all_rigids = flip(all_rigids)
        all_rigids_0 = flip(all_rigids_0)
        all_bb_0_pred = flip(all_bb_0_pred)[:,None]
        traj_out = {
            'seq_traj': all_seqs,
            'prot_traj': all_bb_prots,
            'mask_traj': all_bb_mask,    
            'rigid_traj' : all_rigids,
            'rigid_0_traj' : all_rigids_0,
            'prot_0_traj' : all_bb_0_pred,
        }
        traj_out = tree.map_structure(lambda x: x[:,0], traj_out)
        sample_out = tree.map_structure(lambda x: x[0], sample_out)
        sample_out.update(traj_out)
        return sample_out, init_feats

    def compute_prob(self, v1, v2, eps=1e-8):
        prob = (torch.exp(v1) + eps) / (torch.exp(v1) + torch.exp(v2) + 2*eps)
        return prob.item()

    def log(self, i, kappa_trans, kappa_rots):
        wandb.log({"kappa_trans": kappa_trans.item(),
                   "kappa_rots": kappa_rots.item(),},
                   step=i)
        try:
            wandb.log({"proteus_trans_dlldt": self.val_dict['proteus']['trans']['dlldt'].item(),
                   "framediff_trans_dlldt": self.val_dict['framediff']['trans']['dlldt'].item(),
                   "proteus_trans_ll": self.ll_proteus_trans[:, i + 1].item(),
                   "framediff_trans_ll": self.ll_framediff_trans[:, i + 1].item(),
                   "trans_delta_ll": self.ll_proteus_trans[:, i + 1].item()- self.ll_framediff_trans[:, i + 1].item(),

                   "proteus_rots_dlldt": self.val_dict['proteus']['rots']['dlldt'].item(),
                   "framediff_rots_dlldt": self.val_dict['framediff']['rots']['dlldt'].item(),
                   "proteus_rots_ll": self.ll_proteus_rots[:, i + 1].item(),
                   "framediff_rots_ll": self.ll_framediff_rots[:, i + 1].item(),
                   "rots_delta_ll": self.ll_proteus_rots[:, i + 1].item()- self.ll_framediff_rots[:, i + 1].item(),
                   },
                   step=i)
        except: 
            pass

    def sigma_t(self, t):
        return torch.sqrt(1 - torch.exp(-1*self.r3_diffuser.marginal_b_t(t)))
    
    def sigma_r(self, t):
        max_sigma = self.comp_diff_conf.diffuser.so3.max_sigma
        min_sigma = self.comp_diff_conf.diffuser.so3.min_sigma
        return torch.log(
            t * torch.exp(torch.tensor(max_sigma))
            + (1 - t)
            * torch.exp(torch.tensor(min_sigma))
        )

    def calc_beta_t_trans(self, latents, t): 
        x_t_trans = latents["rigids_t"][:, :, 4:] 
        x_t_trans = self.r3_diffuser._scale(x_t_trans)

        # beta_t = sigma(t)^2 [d/dt log(s_t) - d/dt log(alpha_t)]
        t_ = torch.tensor(t)
        
        def logsigma_t(t):
            return torch.log(self.sigma_t(t))
        
        def logalpha_t(t):
            return -0.5 * self.r3_diffuser.marginal_b_t(t)

        dlog_sigmadt = grad(logsigma_t)(t_)
        dlog_alphadt = grad(logalpha_t)(t_)
        dlog_alphadt_x_trans = dlog_alphadt * x_t_trans 

        beta_t_trans = self.sigma_t(t_)**2 * (dlog_sigmadt - dlog_alphadt) 
        return beta_t_trans, dlog_alphadt_x_trans, dlog_alphadt

    def g_t_trans(self, t):
        return self.r3_diffuser.diffusion_coef(t)

    def f_t_trans(self, latents, t):
        x_t_trans = latents["rigids_t"][:, :, 4:]
        x_t_trans = self.r3_diffuser._scale(x_t_trans)
        return self.r3_diffuser.drift_coef(x_t_trans, t)

    def calc_beta_t_rots(self, t):
        t_ = torch.tensor(t)

        def logsigma_r(t):
            return torch.log(self.sigma_r(t))

        dlog_sigmadt_rots = grad(logsigma_r)(t_)

        beta_t_rots = self.sigma_r(t_)**2 * (dlog_sigmadt_rots)
        return beta_t_rots

    def inference_fn(self):
        init_feats = self.init_feats_proteus
        latents, init_feats = self.latent_mixing(init_feats)
        return latents, init_feats
    
    def save_output(self, init_feats, sample_out, diff_path): 
        
        init_feats = {k: du.move_to_np(v) for k, v in init_feats.items()}

        diff_prot = protein.Protein(
            aatype=sample_out["pred_aatype"][0] if "pred_aatype" in sample_out else init_feats["aatype"][0],
            atom_positions=sample_out["final_atom_positions"],
            atom_mask=sample_out["final_atom_mask"],
            residue_index=init_feats["residue_index"][0],
            chain_index=init_feats["chain_index"][0],
            b_factors=np.tile(1 -init_feats['fixed_mask'][0,..., None], 37) * 100
        )
        open(diff_path,'w').write(protein.to_pdb(diff_prot))
        print("saved file") 

    def _set_t_feats(self, feats, t, t_placeholder):
        feats['t'] = t * t_placeholder
        rot_score_scaling, trans_score_scaling = self.diffuser.score_scaling(t)
        feats['rot_score_scaling'] = rot_score_scaling * t_placeholder
        feats['trans_score_scaling'] = trans_score_scaling * t_placeholder
        return feats 
    
    def proj2manifold(self, x):
        print(x.shape)
        x = x.squeeze(0)
        print(x.shape)
        u, _, vT = torch.linalg.svd(x, full_matrices=False)
        print(u.shape, vT.shape)
        return u @ vT
    
    def _self_conditioning(self, batch, model):
        model_sc = model(batch)
        batch['sc_ca_t'] = model_sc['rigids'][..., 4:]
        return batch
