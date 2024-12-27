import torch
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
import ImageReward as RM
import torchvision

from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import EulerDiscreteScheduler
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import CLIPProcessor, CLIPModel

import wandb
import argparse
import os
from tqdm import tqdm

PATH = "/projects/superdiff/saved_sd_results/"

dtype = torch.float32
device = torch.device("cuda")

sd_model="CompVis/stable-diffusion-v1-4"

vae = AutoencoderKL.from_pretrained(sd_model, subfolder="vae", use_safetensors=True)
tokenizer = CLIPTokenizer.from_pretrained(sd_model, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(
    sd_model, subfolder="text_encoder", use_safetensors=True
)
unet = UNet2DConditionModel.from_pretrained(
    sd_model, subfolder="unet", use_safetensors=True
)

torch_device = torch.device('cuda')
vae.to(torch_device)
text_encoder.to(torch_device)
unet.to(torch_device)

scheduler = EulerDiscreteScheduler.from_pretrained(sd_model, subfolder="scheduler")

@torch.no_grad
def get_image(latents, nrow, ncol):
    image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1).squeeze()
    if len(image.shape) < 4:
        image = image.unsqueeze(0)
    image = (image.permute(0, 2, 3, 1) * 255).to(torch.uint8)
    rows = []
    for row_i in range(nrow):
        row = []
        for col_i in range(ncol):
            i = row_i * nrow + col_i
            row.append(image[i])
        rows.append(torch.hstack(row))
    image = torch.vstack(rows)
    return Image.fromarray(image.cpu().numpy())

@torch.no_grad
def get_image_for_save(latents, nrow, ncol):
    image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1).squeeze()
    image = image.permute(0, 2, 3, 1) * 255  # .to(torch.uint8)
    return image

@torch.no_grad
def get_batch(latents, nrow, ncol):
    image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1).squeeze()
    if len(image.shape) < 4:
        image = image.unsqueeze(0)
    image = (image.permute(0, 2, 3, 1) * 255).to(torch.uint8)
    return image

@torch.no_grad
def get_text_embedding(prompt):
    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return text_encoder(text_input.input_ids.to(torch_device))[0]

@torch.no_grad
def get_vel(t, sigma, latents, embeddings, eps=None, get_div=False):
    v = lambda _x, _e: unet(
        _x / ((sigma**2 + 1) ** 0.5), t, encoder_hidden_states=_e
    ).sample
    embeds = torch.cat(embeddings)
    latent_input = latents
    if get_div:
        with sdpa_kernel(SDPBackend.MATH):
            vel, div = torch.func.jvp(
                v, (latent_input, embeds), (eps, torch.zeros_like(embeds))
            )
            div = -(eps * div).sum((1, 2, 3))
    else:
        vel = v(latent_input, embeds)
        div = torch.zeros([len(embeds)], device=torch_device)
    return vel, div


def compute_clip_score(clip_processor, clip, mixed_samples, args):
    score_min, score_avg, raw_scores = [], [], []
    with torch.no_grad():
        for i in range(args.batch_size):
            inputs = clip_processor(
                text=[args.obj],
                images=mixed_samples[i].unsqueeze(0),
                return_tensors="pt",
                padding=True,
            )
            outputs = clip(**inputs)
            logits_per_image = (
                outputs.logits_per_image
            )  # this is the image-text similarity score
            sim_sd_A = logits_per_image.cpu().item()

            inputs = clip_processor(
                text=[args.bg],
                images=mixed_samples[i].unsqueeze(0),
                return_tensors="pt",
                padding=True,
            )

            outputs = clip(**inputs)
            logits_per_image = (
                outputs.logits_per_image
            )  # this is the image-text similarity score
            sim_sd_B = logits_per_image.cpu().item()

            score_min.append(min(sim_sd_A, sim_sd_B))
            score_avg.append((sim_sd_A + sim_sd_B) / 2)
            raw_scores.append((sim_sd_A, sim_sd_B))
            
    return score_min, score_avg, raw_scores


def compute_image_reward(image_reward, mixed_samples, args):
    score_min, score_avg, raw_scores = [], [], []
    with torch.no_grad():
        for i in range(args.batch_size):
            img = mixed_samples[i].unsqueeze(0)
            img = Image.fromarray(img.squeeze(0).cpu().numpy())
            
            rewards_A = image_reward.score(args.obj, img)
            rewards_B = image_reward.score(args.bg, img)
            
            score_min.append(min(rewards_A, rewards_B))
            score_avg.append((rewards_A + rewards_B) / 2)
            raw_scores.append((rewards_A, rewards_B))

    return score_min, score_avg, raw_scores
    

def get_ll_ode(
    args,
    scheduler,
    images,
    latents_og,
    uncond_embeddings=None,
):
    ll_ode = torch.zeros(
        (args.num_inference_steps + 1, args.batch_size), device=torch_device
    )
    ll_ode_fwd = torch.zeros(
        (args.num_inference_steps + 1, args.batch_size), device=torch_device
    )
    
    rev_sigmas = torch.flip(scheduler.sigmas, dims=(0,))
    
    # forward
    for i, t in enumerate(torch.flip(scheduler.timesteps, dims=(0,))): 
        dsigma = rev_sigmas[i + 1] - rev_sigmas[i]
        sigma = rev_sigmas[i]
        
        eps = torch.randint_like(images, 2, dtype=images.dtype) * 2 - 1
        vf, div = get_vel(t, sigma, images, [uncond_embeddings], eps, True)
        
        images += dsigma * vf
        
        dlog_q = -torch.abs(dsigma) * div
        ll_ode_fwd[i + 1] = ll_ode_fwd[i] + dlog_q
    
    images = images / scheduler.init_noise_sigma
    latents_ode = images.clone().detach()  # for Gaussian log-likelihood computation
    print("latents ODE:", latents_ode[0, 0, 0, :10])
    
    n = latents_ode.shape[1] * latents_ode.shape[2] * latents_ode.shape[3]
    ll_q0_ode = - n / 2 * (torch.log(torch.tensor(2 * np.pi)) - torch.log(scheduler.init_noise_sigma**2)) * torch.ones((args.batch_size), device=torch_device)
    ll_q0_ode -= 1 / scheduler.init_noise_sigma**2 * ((images * scheduler.init_noise_sigma) ** 2).sum((1, 2, 3))
    
    latents_ode = latents_ode * scheduler.init_noise_sigma

    # reverse
    for i, t in enumerate(scheduler.timesteps): 
        dsigma = scheduler.sigmas[i + 1] - scheduler.sigmas[i]
        sigma = scheduler.sigmas[i]
        eps = torch.randint_like(latents_ode, 2, dtype=latents_ode.dtype) * 2 - 1

        vf, div = get_vel(t, sigma, latents_ode, [uncond_embeddings], eps, True)
        latents_ode += dsigma * vf

        dlog_q = -torch.abs(dsigma) * div
        ll_ode[i + 1] = ll_ode[i] + dlog_q

    ll_q1_ode_adj = ll_ode[-1] + ll_q0_ode
    print("ODE-FWD (no q_0) w/o guidance log-likelihoods:", ll_ode_fwd[-1])
    print("ODE (no q_0) w/o guidance log-likelihoods:", ll_ode[-1])
    print("ODE (yes q_0) w/o guidance log-likelihoods:", ll_q1_ode_adj)
    print("log(q_0):", ll_q0_ode)
    
    return ll_q1_ode_adj, ll_ode[-1]
    
    
def get_ll_ode_guidance(
    args,
    scheduler,
    images,
    latents_og,
    uncond_embeddings=None,
    obj_embeddings=None,
    bg_embeddings=None,
):
    
    ll_ode_obj = torch.zeros((args.num_inference_steps+1, args.batch_size), device=torch_device)
    ll_ode_bg = torch.zeros((args.num_inference_steps+1, args.batch_size), device=torch_device)

    rev_sigmas = torch.flip(scheduler.sigmas, dims=(0,))
    for i, t in enumerate(torch.flip(scheduler.timesteps, dims=(0,))):  # forward
        dsigma = rev_sigmas[i + 1] - rev_sigmas[i]
        sigma = rev_sigmas[i]
        
        if obj_embeddings is not None:
            vel_obj, _ = get_vel(t, sigma, images, [obj_embeddings], None, False)
        if bg_embeddings is not None:
            vel_bg, _ = get_vel(t, sigma, images, [bg_embeddings], None, False)
        if uncond_embeddings is not None:
            vel_uncond, _ = get_vel(t, sigma, images, [uncond_embeddings], None, False)
        
        vf = vel_uncond + args.guidance_scale*(vel_obj - vel_uncond)
        images += dsigma * vf
    
    images = images / scheduler.init_noise_sigma
    latents_ode = images.clone().detach()  # for Gaussian log-likelihood computation
    
    n = latents_ode.shape[1] * latents_ode.shape[2] * latents_ode.shape[3]
    ll_q0_ode = - n / 2 * (torch.log(torch.tensor(2 * np.pi)) - torch.log(scheduler.init_noise_sigma**2)) * torch.ones((args.batch_size), device=torch_device)
    ll_q0_ode -= 1 / scheduler.init_noise_sigma**2 * ((latents_ode * scheduler.init_noise_sigma) ** 2).sum((1, 2, 3))
    
    latents_ode = latents_ode * scheduler.init_noise_sigma
    
    for i, t in enumerate(scheduler.timesteps):  # reverse
        dsigma = scheduler.sigmas[i + 1] - scheduler.sigmas[i]
        sigma = scheduler.sigmas[i]
        eps = torch.randint_like(latents_ode, 2, dtype=latents_ode.dtype) * 2 - 1
        
        if obj_embeddings is not None:
            vel_obj, div_obj = get_vel(t, sigma, latents_ode, [obj_embeddings], eps, True)
        if bg_embeddings is not None:
            vel_bg, div_bg = get_vel(t, sigma, latents_ode, [bg_embeddings], eps, True)
        if uncond_embeddings is not None:
            vel_uncond, _ = get_vel(t, sigma, latents_ode, [uncond_embeddings], eps, False)
        
        vf = vel_uncond + args.guidance_scale*(vel_obj - vel_uncond)
        latents_ode += dsigma * vf
                
        if obj_embeddings is not None:
            dlog_q_obj = -torch.abs(dsigma) * div_obj
            dlog_q_obj += -torch.abs(dsigma) * ((-vel_obj / sigma) * (vel_obj-vf)).sum((1,2,3)) # recall v(t) = -sigma(t) * s(t)
            ll_ode_obj[i+1] = ll_ode_obj[i] + dlog_q_obj
        if bg_embeddings is not None:
            ll_ode_bg[i+1] = ll_ode_bg[i] + dsigma*(div_bg - ((-vel_bg/sigma)*(vel_bg-vf)).sum((1,2,3)))
        
    ll_q1_ode_obj_adj = ll_ode_obj[-1] + ll_q0_ode
    print("ODE (no q_0) w/ guidance log-likelihoods:", ll_ode_obj[-1])
    print("ODE (yes q_0) w/ guidance log-likelihoods:", ll_q1_ode_obj_adj)
    print("log(q_0):", ll_q0_ode)
    
    return ll_q1_ode_obj_adj, ll_ode_obj[-1]
    
def run(args):
    if args.method == "sd_ab":
        prompt_for_sd = args.obj + " that looks like " + args.bg
        obj_embeddings = get_text_embedding([prompt_for_sd] * args.batch_size)
    elif args.method == "sd_ab_or":
        prompt_for_sd = args.obj + " or " + args.bg
        obj_embeddings = get_text_embedding([prompt_for_sd] * args.batch_size)
    elif args.method == "sd_a":
        prompt_for_sd = args.obj
        obj_embeddings = get_text_embedding([prompt_for_sd] * args.batch_size)
    elif args.method == "sd_ba":
        prompt_for_sd = args.bg + " that looks like " + args.obj
        obj_embeddings = get_text_embedding([prompt_for_sd] * args.batch_size)
    elif args.method == "sd_ba_or":
        prompt_for_sd = args.bg + " or " + args.obj
        obj_embeddings = get_text_embedding([prompt_for_sd] * args.batch_size)
    elif args.method == "sd_b":
        prompt_for_sd = args.bg
        obj_embeddings = get_text_embedding([prompt_for_sd] * args.batch_size)
    elif args.method == "and" or args.method == "or" or args.method == "and_ode":
        lift = 0.0
        kappa = 0.5*torch.ones((args.num_inference_steps+1,args.batch_size), device=torch_device)
        obj_prompt = [args.obj]
        bg_prompt = [args.bg]
        obj_embeddings = get_text_embedding(obj_prompt * args.batch_size)
        bg_embeddings = get_text_embedding(bg_prompt * args.batch_size)
    elif args.method == "avg":
        kappa = 0.5
        obj_prompt = [args.obj]
        bg_prompt = [args.bg]
        obj_embeddings = get_text_embedding(obj_prompt * args.batch_size)
        bg_embeddings = get_text_embedding(bg_prompt * args.batch_size)
    else:
        raise ValueError("Method not recognized")
    
    uncond_embeddings = get_text_embedding([""] * args.batch_size)
    
    prompt_for_save = (
            args.obj.replace(" ", "_") + "_" + "and" + "_" + args.bg.replace(" ", "_")
        )
    print("file dir save name:", prompt_for_save)
            
    generator = torch.cuda.manual_seed(args.seed)  # Seed generator to create the initial latent noise
    latents = torch.randn(
        (args.batch_size, unet.config.in_channels, args.height // 8, args.width // 8),
        generator=generator,
        device=torch_device,
    )
    
    latents_og = latents.clone().detach()
    latents_uncond_og = latents.clone().detach()
    
    scheduler.set_timesteps(args.num_inference_steps)
    latents = latents * scheduler.init_noise_sigma
    
    latents_uncond = latents.clone().detach()

    # run inference
    print("Running inference...")
    start_time = time.time()  # Record the start time

    ll_obj = torch.ones((args.num_inference_steps+1,args.batch_size), device=torch_device)
    ll_bg = torch.ones((args.num_inference_steps+1,args.batch_size), device=torch_device)
    ll_uncond = torch.ones((args.num_inference_steps+1,args.batch_size), device=torch_device)
    for i, t in tqdm(enumerate(scheduler.timesteps), colour="MAGENTA"):
        dsigma = scheduler.sigmas[i + 1] - scheduler.sigmas[i]
        sigma = scheduler.sigmas[i]
        vel_obj, _ = get_vel(t, sigma, latents, [obj_embeddings])
        vel_uncond, _ = get_vel(t, sigma, latents, [uncond_embeddings])
        
        if args.method in ["sd_ab", "sd_ba", "sd_ab_or", "sd_ba_or", "sd_a", "sd_b"]:
            zs = torch.randn_like(latents)
            
            # guidance generation
            vf = vel_uncond + args.guidance_scale * (vel_obj - vel_uncond)
            dx = 2*dsigma*vf + torch.sqrt(2*torch.abs(dsigma)*sigma)*zs
            latents += dx
            
            # unconditional generation
            vel_uncond_only, _ = get_vel(t, sigma, latents_uncond, [uncond_embeddings])
            dx_uncond = 2*dsigma*vel_uncond_only + torch.sqrt(2*torch.abs(dsigma)*sigma)*zs
            latents_uncond += dx_uncond
            
            # compute SDE log-likelihood
            ll_obj[i+1] = ll_obj[i] + (-torch.abs(dsigma)/sigma*(vel_obj)**2 - (dx*(vel_obj/sigma))).sum((1,2,3))
            ll_bg[i+1] = ll_obj[i+1]
            
            ll_uncond[i+1] = ll_uncond[i] + (-torch.abs(dsigma)/sigma*(vel_uncond_only)**2 - (dx*(vel_uncond_only/sigma))).sum((1,2,3))
            
            wandb.log({"iter": i, "ll_obj": ll_obj[i+1].mean(), "ll_bg": ll_bg[i+1].mean()})
        elif args.method == "and_ode":

            eps = torch.randint_like(latents, 2, dtype=latents.dtype)*2-1
            vel_obj, dlog_obj = get_vel(t, sigma, latents, [obj_embeddings], eps, True)
            vel_bg, dlog_bg = get_vel(t, sigma, latents, [bg_embeddings], eps, True)
            vel_uncond, _ = get_vel(t, sigma, latents, [uncond_embeddings], eps, False)

            kappa[i+1] = sigma*(dlog_obj - dlog_bg)+((vel_obj-vel_bg)*(vel_obj+vel_bg)).sum((1,2,3)) + lift/dsigma*sigma/args.num_inference_steps
            kappa[i+1] += -((vel_obj-vel_bg)*(vel_uncond + args.guidance_scale*(vel_bg-vel_uncond))).sum((1,2,3))
            kappa[i+1] /= args.guidance_scale*((vel_obj-vel_bg)**2).sum((1,2,3))

            vf = vel_uncond + args.guidance_scale*((vel_bg - vel_uncond) + kappa[i+1][:,None,None,None]*(vel_obj-vel_bg))
            latents += dsigma*vf
            ll_obj[i+1] = ll_obj[i] + dsigma*(dlog_obj - ((-vel_obj/sigma)*(vel_obj-vf)).sum((1,2,3)))
            ll_bg[i+1] = ll_bg[i] + dsigma*(dlog_bg - ((-vel_bg/sigma)*(vel_bg-vf)).sum((1,2,3)))

        elif args.method == "and" or args.method == "or":
            vel_bg, _ = get_vel(t, sigma, latents, [bg_embeddings])
            noise = torch.sqrt(2*torch.abs(dsigma)*sigma)*torch.randn_like(latents)
            
            if args.method == "and":
                dx_ind = 2*dsigma*(vel_uncond + args.guidance_scale*(vel_bg - vel_uncond)) + noise
                kappa[i+1] = (torch.abs(dsigma)*(vel_bg-vel_obj)*(vel_bg+vel_obj)).sum((1,2,3))-(dx_ind*((vel_obj-vel_bg))).sum((1,2,3)) + sigma*lift/args.num_inference_steps
                kappa[i+1] /= 2*dsigma*args.guidance_scale*((vel_obj-vel_bg)**2).sum((1,2,3))
            elif args.method == "or":
                kappa[i+1] = torch.softmax(torch.stack([args.T*(ll_obj[i]+args.logp), args.T*(ll_bg[i])]), 0)[0]
            
            vf = vel_uncond + args.guidance_scale*((vel_bg - vel_uncond) + kappa[i+1][:,None,None,None]*(vel_obj-vel_bg))
            dx = 2*dsigma*vf + noise
            latents += dx
            
            if args.method == "and":
                ll_obj[i+1] = ll_obj[i] + (-torch.abs(dsigma)/sigma*(vel_obj)**2 - (dx*(vel_obj/sigma))).sum((1,2,3))
                ll_bg[i+1] = ll_bg[i] + (-torch.abs(dsigma)/sigma*(vel_bg)**2 - (dx*(vel_bg/sigma))).sum((1,2,3)) 
            elif args.method == "or":
                ll_obj[i+1] = ll_obj[i] - (vel_obj*(dx + dsigma*vel_obj)/sigma).sum((1,2,3))
                ll_bg[i+1] = ll_bg[i] - (vel_bg*(dx + dsigma*vel_bg)/sigma).sum((1,2,3))
                 
            wandb.log({"iter": i, "kappa": kappa[i+1].mean(), "ll_obj": ll_obj[i+1].mean(), "ll_bg": ll_bg[i+1].mean()})
            
        elif args.method == "avg": 
            vel_bg, _ = get_vel(t, sigma, latents, [bg_embeddings])
            vf = vel_uncond + args.guidance_scale*((vel_bg - vel_uncond) + kappa*(vel_obj-vel_bg))
            dx = 2*dsigma*vf + torch.sqrt(2*torch.abs(dsigma)*sigma)*torch.randn_like(latents)
            latents += dx
            
            ll_obj[i+1] = ll_obj[i] + (-torch.abs(dsigma)/sigma*(vel_obj)**2 - (dx*(vel_obj/sigma))).sum((1,2,3))
            ll_bg[i+1] = ll_bg[i] + (-torch.abs(dsigma)/sigma*(vel_bg)**2 - (dx*(vel_bg/sigma))).sum((1,2,3))
            
            wandb.log({"iter": i, "kappa": kappa, "ll_obj": ll_obj[i+1].mean(), "ll_bg": ll_bg[i+1].mean()})

        elif args.method == "avg_ode": 
            vel_bg, _ = get_vel(t, sigma, latents, [bg_embeddings])
            vf = vel_uncond + args.guidance_scale*((vel_bg - vel_uncond) + kappa*(vel_obj-vel_bg))
            dx = dsigma*vf 
            latents += dx
            
            
            wandb.log({"iter": i, "kappa": kappa})
   
   
        else:
            raise ValueError("Method not recognized")
    
    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time  # Calculate the time taken
    print(f"The function took {execution_time:.4f} seconds to run.")
    
    mixed_samples = get_batch(latents, 1, args.batch_size)
    print("Inference done.")
    
    # compute CLIP score
    print("Computing CLIP score...")
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_score_min, clip_score_avg, clip_raw_scores = compute_clip_score(clip_processor, clip, mixed_samples, args)  
      
    print("\n === CLIP similarity score ===")
    mean_clip_score_min = sum(clip_score_min) / args.batch_size
    mean_clip_score_avg = sum(clip_score_avg) / args.batch_size
    print("min:", mean_clip_score_min, "avg:", mean_clip_score_avg)
    print("CLIP score computed.\n")
        
    # compute Image Reward
    image_reward = RM.load("ImageReward-v1.0")#, download_root='/projects/superdiff/')
    image_reward_score_min, image_reward_score_avg, ir_raw_scores = compute_image_reward(
        image_reward, mixed_samples, args
    )
    
    print("\n === ImageReward score ===")
    mean_image_reward_score_min = sum(image_reward_score_min) / args.batch_size
    mean_image_reward_score_avg = sum(image_reward_score_avg) / args.batch_size
    print("min:", mean_image_reward_score_min, "avg:", mean_image_reward_score_avg)
    print("ImageReward score computed.\n")
    
    if args.method == "and" or args.method == "avg":
        wandb.log(
            {
                "clip_raw_scores": str(clip_raw_scores),
                "ir_raw_scores": str(ir_raw_scores),
                "min_clip_score_batch_mean": mean_clip_score_min,
                "min_IR_score_batch_mean": mean_image_reward_score_min,
                "avg_clip_score_batch_mean": mean_clip_score_avg,
                "avg_IR_score_batch_mean": mean_image_reward_score_avg,
                "ll_obj_batch_mean": ll_obj[-1].mean(),
                "ll_bg_batch_mean": ll_bg[-1].mean(),
            }
        )
    else:
        wandb.log(
            {
                "clip_raw_scores": str(clip_raw_scores),
                "ir_raw_scores": str(ir_raw_scores),
                "min_clip_score_batch_mean": mean_clip_score_min,
                "min_IR_score_batch_mean": mean_image_reward_score_min,
                "avg_clip_score_batch_mean": mean_clip_score_avg,
                "avg_IR_score_batch_mean": mean_image_reward_score_avg,
            }
        )
    
    # save images
    subdir = prompt_for_save

    if args.T != 1:
        args.method = f"{args.method}_T{args.T}"

    if not os.path.exists(PATH + args.method + "/" + subdir):
        os.makedirs(PATH + args.method + "/" + subdir)
    for i in range(latents.shape[0]):
        img_mixed_to_save = get_image(latents[i].unsqueeze(0), 1, 1)
        img_mixed_to_save.save(PATH + args.method + "/" + subdir + f"/{i}.png")

    # save metrics
    # Each key is the image name, and the value is a dictionary of metrics
    metrics_data = {}
    for i in range(args.batch_size):
        metrics_data[f"image_{i}"] = {
            "clip_raw_score_1": clip_raw_scores[i][0],
            "clip_raw_score_2": clip_raw_scores[i][1],
            "ir_raw_score_1": ir_raw_scores[i][0],
            "ir_raw_score_2": ir_raw_scores[i][1],
            "min_clip": clip_score_min[i],
            "avg_clip": clip_score_avg[i],
            "min_ir": image_reward_score_min[i],
            "avg_ir": image_reward_score_avg[i],
        }
        
    df = pd.DataFrame.from_dict(metrics_data, orient="index")

    csv_filename = f"metrics_{args.method}_{prompt_for_save}.csv"
    if not os.path.exists(PATH + f"metrics_{args.method}/"):
        os.makedirs(PATH + f"metrics_{args.method}/")
    df.to_csv(PATH + f"metrics_{args.method}/" + csv_filename, index_label="Image")
       
def main():
    # arguments
    parser = argparse.ArgumentParser(description="Your script description here")
    parser.add_argument('--num_inference_steps', type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--T", type=float, default=1, help="temperature for OR")
    parser.add_argument("--logp", type=float, default=0, help="bias for OR")
    parser.add_argument("--guidance_scale", type=int, default=7.5)
    parser.add_argument("--obj", type=str, default="a cat")
    parser.add_argument("--bg", type=str, default="a dog")
    parser.add_argument("--method", type=str, default="and", choices=["sd_ab", "sd_ba", "avg", "and", "avg_ode", "or", "sd_ab_or", "sd_ba_or", "sd_a", "sd_b", "and_ode"])
    parser.add_argument("--compare_density_est", type=bool, default=False)
    parser.add_argument("--tags", type=str, default=["score_test", "time_measurement"])
    args = parser.parse_args()
    
    wandb.init(
        project="superdiff_imgs", 
        config=args,
        tags=args.tags,
        )
    
    # run script
    print("Script is running with the provided arguments.\n")
    print(args)
    run(args)
    

if __name__ == "__main__":
    main()
