from typing import Dict
from pathlib import Path
import os, sys
import numpy as np
import random
import logging
from hydra import compose, initialize
import tree
import GPUtil

from omegaconf import DictConfig, OmegaConf
import torch
import hydra
import wandb

# Proteus imports
from proteus_experiments.inference_se3_diffusion import Sampler as ProteusSampler
from se3diff_experiments.inference_se3_diffusion import Sampler as FrameDiffSampler

from tqdm import tqdm
from composition import CompositionDiffusion
sys.path.append("../evaluation")
from run_self_consistency import SelfConsistency


class Sampler:
    def __init__(
            self,
            conf: DictConfig,
            conf_overrides: Dict=None,
            seed: int=0

        ):
        """Initialize sampler.

        Args:
            conf: inference config.
            gpu_id: GPU device ID.
            conf_overrides: Dict of fields to override with new values.
            """
        self._log = logging.getLogger(__name__)

        OmegaConf.set_struct(conf, False)
        self._conf = conf
        self._infer_conf = conf.inference

        self._rng = np.random.default_rng(self._infer_conf.seed)
        self.seed = seed

        self.sample_length = self._infer_conf.sample_length

        if torch.cuda.is_available():
            if self._infer_conf.gpu_id is None:
                available_gpus = ''.join(
                    [str(x) for x in GPUtil.getAvailable(
                        order='memory', limit = 8)])
                self.device = f'cuda:{available_gpus[0]}'
            else:
                self.device = f'cuda:{self._infer_conf.gpu_id}'
        else:
            self.device = 'cpu'

        self._log.info(f'Using device: {self.device}')

        self.model_info = None
        self.model2loader = {'proteus': ProteusSampler,
                        "framediff": FrameDiffSampler}

        self.sc_evaluator = None

    def load_models(self):
        if self.model_info is None:
            self.model_info = {}
            for model_name in self._infer_conf.models:
                path_ = Path(self._infer_conf.models[model_name])
                hydra.core.global_hydra.GlobalHydra.instance().clear()
                with initialize(version_base=None, config_path=str(path_.parent)):
                    conf = compose(config_name=path_.stem)
                sampler = self.model2loader[model_name](conf, save_outputs=False)
                model = self.initialize_model(model_name, sampler)
                self.model_info[model_name] = {"model": model, "conf": conf}
            print("done loading models") 
        
        #exit()

        
    def load_sc_obj(self):
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        with initialize(version_base=None, config_path="../evaluation/sc_config"):
            conf = compose(config_name="inference")
        self.sc_evaluator = SelfConsistency(conf, input_dir=self._infer_conf.save_path)
        print("loaded self-consistency object")

    def _set_seed(self, seed):
        print(f"setting seed.........{seed}")
        self.seed = seed
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

    def reset_init_feats(self):
        for model_name in self._infer_conf.models:
            path_ = Path(self._infer_conf.models[model_name])
            hydra.core.global_hydra.GlobalHydra.instance().clear()
            with initialize(version_base=None, config_path=str(path_.parent)):
                conf = compose(config_name=path_.stem)
            sampler = self.model2loader[model_name](conf, save_outputs=False)
            init_feats = self.initialize_feats(model_name, sampler)
            self.model_info[model_name]["init_feats"] = init_feats
        print("done resetting feats") 

    def initialize_model(self, model_name, sampler):
        def framediff(sampler):
            print("loading framediff model")
            model = sampler.exp.model
            return model
        def proteus(sampler):
            print("loading proteus model")
            model = sampler._fold_module.structure_model
            return model
        model = locals()[model_name](sampler)#(self, f"init_model_and_feats.{model_name}")(sampler)
        return model

    def initialize_feats(self, model_name, sampler):
        def framediff(sampler):
            print("loading framediff init feats")
            res_mask = np.ones(self.sample_length)
            fixed_mask = np.zeros_like(res_mask)
            
            # Initialize data
            ref_sample = sampler.diffuser.sample_ref(
                n_samples=self.sample_length,
                as_tensor_7=True,
            )
            res_idx = torch.arange(1, self.sample_length+1)
            init_feats = {
                'res_mask': res_mask,
                'seq_idx': res_idx,
                'fixed_mask': fixed_mask,
                'torsion_angles_sin_cos': np.zeros((self.sample_length, 7, 2)),
                'sc_ca_t': np.zeros((self.sample_length, 3)),
                **ref_sample,
            }
            # Add batch dimension and move to GPU.
            init_feats = tree.map_structure(
                lambda x: x if torch.is_tensor(x) else torch.tensor(x), init_feats)
            init_feats = tree.map_structure(
                lambda x: x[None].to(self.device), init_feats)
            
            return init_feats
        
        def proteus(sampler):
            print("loading proteus init feats")
            init_feats = sampler._fold_module.init_feat(contigs=f"{self.sample_length}-{self.sample_length}",
                                                        ref_feats=None,
                                                        hotspot=None)
            ## ADDDED THIS SO IT RUNS WITH FRAMEDIFF
            #res_idx = torch.arange(1, self.sample_length+1)
            framediff_feats = {'seq_idx': torch.arange(1, self.sample_length+1),
                               'sc_ca_t': torch.tensor(np.zeros((self.sample_length, 3)))}
            framediff_feats = tree.map_structure(
                lambda x: x if torch.is_tensor(x) else torch.tensor(x), framediff_feats)
            framediff_feats = tree.map_structure(
                lambda x: x[None].to(self.device), framediff_feats)
            init_feats.update(framediff_feats)

            return init_feats
        
        init_feats = locals()[model_name](sampler)#(self, f"init_model_and_feats.{model_name}")(sampler)
        return init_feats

def run_one(sampler, global_conf, i):

    assert "../superdiff/generated_proteins/" in global_conf.inference.save_path
    root_name = global_conf.inference.save_path.split("/")[3] + f"_SEED{i}"
    tags = [f"SEED{i}", f'temp{global_conf.inference.temp_trans}', 
            f"seqlen{global_conf.inference.sample_length}", f"logp{global_conf.inference.logp_trans}"]
    
    save_path = sampler._infer_conf.save_path
    os.makedirs(save_path, exist_ok=True)
    save_path = Path(save_path)
    if not os.path.exists(str(save_path.parent) + "/config.yaml"):
       OmegaConf.save(global_conf, str(save_path.parent) + "/config.yaml")
    save_path = str(save_path)
    save_path += f"/sample_{i}.pdb"
    # generated_proteins/proteus_sde_500t/length_100/self_consistency/sample_2/sc_results.csv
    csv_path = save_path.replace(f"model_outputs/sample_{i}.pdb", f"self_consistency/sample_{i}/sc_results.csv")
    if os.path.exists(save_path) and os.path.exists(csv_path):
        print("path already exists; exiting now!", save_path, csv_path)
        exit()

    wandb.init(project='superdiffusion', name=root_name,
            tags=tags)
    #wandb.init(settings=wandb.Settings(code_dir="."))
    wandb.run.log_code(".")
    wandb.config.update(global_conf)


    sampler.load_models()
    print(f"on seed................{i}")
    sampler._set_seed(i)
    sampler.reset_init_feats()

    superdiff = CompositionDiffusion(comp_diff_conf=global_conf, models=sampler.model_info)
    #
    ## test inference function
    print("\n Running inference_fn ... \n")
    latents, init_feats = superdiff.inference_fn()

    print(f"............saving to {save_path}")
    superdiff.save_output(init_feats, latents, save_path)
    print("... Composition runs! \n")
    if sampler._infer_conf.self_consistency.enable:
        sampler.model_info = None
        sampler.load_sc_obj()
        sc_results = sampler.sc_evaluator.run_one_sc(save_path)
        print({"min_rmsd": sc_results['rmsd'].min()})
        wandb.log({"min_rmsd": sc_results['rmsd'].min()}, step=0)


def run_series(sampler, global_conf):
    for i in tqdm(range(50), colour="MAGENTA"): 
        sampler.load_models()
        run_one(sampler, global_conf, i)

@hydra.main(version_base=None, config_path="config", config_name="composition")
def main(global_conf: DictConfig) -> None:
    seed = global_conf.inference.seed
    sampler = Sampler(global_conf)

    print("seed", global_conf.inference.seed)
    
    if seed is None:
        run_series(sampler, global_conf)
    else:
        run_one(sampler, global_conf, i=seed)

if __name__ == "__main__":
    main()
