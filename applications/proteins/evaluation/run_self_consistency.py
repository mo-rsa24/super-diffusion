"""Script for running inference and sampling.

Sample command:
> python scripts/run_inference.py

"""

import os
import time
import tree
import numpy as np
import hydra
import torch
import subprocess
import logging
import pandas as pd
import shutil
from datetime import datetime
from biotite.sequence.io import fasta
import GPUtil
from typing import Optional
from pathlib import Path

from analysis import utils as au
from analysis import metrics
from data import utils as du
from data import residue_constants
from typing import Dict
from omegaconf import DictConfig, OmegaConf
import esm
from tqdm import tqdm


class SelfConsistency:

    def __init__(
            self,
            conf: DictConfig,
            input_dir: str=None
        ):
        """Initialize sampler.

        Args:
            conf: inference config.
            gpu_id: GPU device ID.
        """
        self._log = logging.getLogger(__name__)

        # Remove static type checking.
        OmegaConf.set_struct(conf, False) # yaml thing for configs

        # Prepare configs.
        self._conf = conf
        self._infer_conf = conf.inference
        self._sample_conf = self._infer_conf.samples

        self._rng = np.random.default_rng(self._infer_conf.seed)

        # Set model hub directory for ESMFold.
        torch.hub.set_dir(self._infer_conf.pt_hub_dir)

        # Set-up accelerator
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

        # Set-up directories
        #output_dir =self._infer_conf.output_dir
        if input_dir is not None:
            self._input_dir = input_dir
        else:
        #if "input_dir" in self._infer_conf:
            self._input_dir = self._infer_conf.input_dir
        #elif "save_path" in self._infer_conf:
        #    self._input_dir = self._infer_conf.save_path

        assert self._input_dir != "" or self._input_dir != None, "You must provide a path containing generated protein structures!"

        self._output_dir = str(Path(self._input_dir).parent / 'self_consistency') 

        os.makedirs(self._output_dir, exist_ok=True)
        self._log.info(f'Saving results to {self._output_dir}')
        self._pmpnn_dir = self._infer_conf.pmpnn_dir

        # Load models and experiment
        self._folding_model = esm.pretrained.esmfold_v1().eval()
        self._folding_model = self._folding_model.to(self.device)

    def run_one_sc(self, pdb_path):
        print("pdb_path", pdb_path)
        if type(pdb_path) == str:
            pdb_path = Path(pdb_path)
        sc_output_dir = Path(self._output_dir) / pdb_path.stem
        print("sc_output_dir", str(sc_output_dir))

        pdb_path = str(pdb_path)
        sc_output_dir = str(sc_output_dir)
        if os.path.exists(os.path.join(sc_output_dir, 'sc_results.csv')): 
            print(f"already done {sc_output_dir}, skipping....")
            return 

        os.makedirs(sc_output_dir, exist_ok=True)
        config_path = os.path.join(sc_output_dir, 'inference_conf.yaml')
        with open(config_path, 'w') as f:
            OmegaConf.save(config=self._conf, f=f)
        self._log.info(f'Saving inference config to {config_path}')

        shutil.copy(pdb_path, os.path.join(
            sc_output_dir, os.path.basename(pdb_path)))
        mpnn_results = self.run_self_consistency(
            sc_output_dir,
            pdb_path,
            motif_mask=None
        )
        return mpnn_results
        self._log.info(f'Done sample : {pdb_path} | saved in : {sc_output_dir}')

    def main(self):
        """
        Run self-consistency
        
        """
        file_idx = -1
        for rootdir, _, filenames in os.walk(self._input_dir):
            for _, f in tqdm(enumerate(sorted(filenames)), colour='magenta'):
                if ".pdb" not in f: continue
                if "traj" in f: continue
                file_idx += 1 
                if self._infer_conf.sample_num != -1:
                    if file_idx != self._infer_conf.sample_num: 
                        continue
                pdb_path = os.path.join(rootdir, f)
                self.run_one_sc(pdb_path)
                #pdb_path = Path(os.path.join(rootdir, f))
                #sc_output_dir = Path(self._output_dir) / pdb_path.stem

                #pdb_path = str(pdb_path)
                #print(pdb_path)
                #sc_output_dir = str(sc_output_dir)
                #if os.path.exists(os.path.join(sc_output_dir, 'sc_results.csv')): 
                #    print(f"already done {sc_output_dir}, skipping....")
                #    continue

                #os.makedirs(sc_output_dir, exist_ok=True)
                #config_path = os.path.join(sc_output_dir, 'inference_conf.yaml')
                #with open(config_path, 'w') as f:
                #    OmegaConf.save(config=self._conf, f=f)
                #self._log.info(f'Saving inference config to {config_path}')

                #shutil.copy(pdb_path, os.path.join(
                #    sc_output_dir, os.path.basename(pdb_path)))
                #_ = self.run_self_consistency(
                #    sc_output_dir,
                #    pdb_path,
                #    motif_mask=None
                #)
                #self._log.info(f'Done sample : {pdb_path} | saved in : {sc_output_dir}')

    def save_traj(
            self,
            bb_prot_traj: np.ndarray,
            x0_traj: np.ndarray,
            diffuse_mask: np.ndarray,
            output_dir: str
        ):
        """Writes final sample and reverse diffusion trajectory.

        Args:
            bb_prot_traj: [T, N, 37, 3] atom37 sampled diffusion states.
                T is number of time steps. First time step is t=eps,
                i.e. bb_prot_traj[0] is the final sample after reverse diffusion.
                N is number of residues.
            x0_traj: [T, N, 3] x_0 predictions of C-alpha at each time step.
            aatype: [T, N, 21] amino acid probability vector trajectory.
            res_mask: [N] residue mask.
            diffuse_mask: [N] which residues are diffused.
            output_dir: where to save samples.

        Returns:
            Dictionary with paths to saved samples.
                'sample_path': PDB file of final state of reverse trajectory.
                'traj_path': PDB file os all intermediate diffused states.
                'x0_traj_path': PDB file of C-alpha x_0 predictions at each state.
            b_factors are set to 100 for diffused residues and 0 for motif
            residues if there are any.
        """

        # Write sample.
        diffuse_mask = diffuse_mask.astype(bool)
        sample_path = os.path.join(output_dir, 'sample')
        prot_traj_path = os.path.join(output_dir, 'bb_traj')
        x0_traj_path = os.path.join(output_dir, 'x0_traj')

        # Use b-factors to specify which residues are diffused.
        b_factors = np.tile((diffuse_mask * 100)[:, None], (1, 37))

        sample_path = au.write_prot_to_pdb(
            bb_prot_traj[0],
            sample_path,
            b_factors=b_factors
        )
        prot_traj_path = au.write_prot_to_pdb(
            bb_prot_traj,
            prot_traj_path,
            b_factors=b_factors
        )
        x0_traj_path = au.write_prot_to_pdb(
            x0_traj,
            x0_traj_path,
            b_factors=b_factors
        )
        return {
            'sample_path': sample_path,
            'traj_path': prot_traj_path,
            'x0_traj_path': x0_traj_path,
        }

    def run_self_consistency(
            self,
            decoy_pdb_dir: str,
            reference_pdb_path: str,
            motif_mask: Optional[np.ndarray]=None):
        """Run self-consistency on design proteins against reference protein.
        
        Args:
            decoy_pdb_dir: directory where designed protein files are stored.
            reference_pdb_path: path to reference protein file
            motif_mask: Optional mask of which residues are the motif.

        Returns:
            Writes ProteinMPNN outputs to decoy_pdb_dir/seqs
            Writes ESMFold outputs to decoy_pdb_dir/esmf
            Writes results in decoy_pdb_dir/sc_results.csv
        """

        # Run PorteinMPNN
        output_path = os.path.join(decoy_pdb_dir, "parsed_pdbs.jsonl")
        process = subprocess.Popen([
            'python',
            f'{self._pmpnn_dir}/helper_scripts/parse_multiple_chains.py',
            f'--input_path={decoy_pdb_dir}',
            f'--output_path={output_path}',
        ])
        _ = process.wait()
        num_tries = 0
        ret = -1
        pmpnn_args = [
            'python',
            f'{self._pmpnn_dir}/protein_mpnn_run.py',
            '--out_folder',
            decoy_pdb_dir,
            '--jsonl_path',
            output_path,
            '--num_seq_per_target',
            str(self._sample_conf.seq_per_sample),
            '--sampling_temp',
            '0.1',
            '--seed',
            '38',
            '--batch_size',
            '1',
        ]
        #if self._infer_conf.gpu_id is not None:
        #    pmpnn_args.append('--device')
        #    pmpnn_args.append(str(self._infer_conf.gpu_id))
        while ret < 0:
            try:
                process = subprocess.Popen(
                    pmpnn_args,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT
                )
                ret = process.wait()
                ret +=1
            except Exception as e:
                num_tries += 1
                self._log.info(f'Failed ProteinMPNN. Attempt {num_tries}/5')
                torch.cuda.empty_cache()
                if num_tries > 4:
                    raise e
        mpnn_fasta_path = os.path.join(
            decoy_pdb_dir,
            'seqs',
            os.path.basename(reference_pdb_path).replace('.pdb', '.fa')
        )

        # Run ESMFold on each ProteinMPNN sequence and calculate metrics.
        mpnn_results = {
            'tm_score': [],
            'sample_path': [],
            'header': [],
            'sequence': [],
            'rmsd': [],
        }
        if motif_mask is not None:
            # Only calculate motif RMSD if mask is specified.
            mpnn_results['motif_rmsd'] = []
        esmf_dir = os.path.join(decoy_pdb_dir, 'esmf')
        os.makedirs(esmf_dir, exist_ok=True)
        fasta_seqs = fasta.FastaFile.read(mpnn_fasta_path)
        sample_feats = du.parse_pdb_feats('sample', reference_pdb_path)
        for i, (header, string) in enumerate(fasta_seqs.items()):

            # Run ESMFold
            esmf_sample_path = os.path.join(esmf_dir, f'sample_{i}.pdb')
            _ = self.run_folding(string, esmf_sample_path)
            esmf_feats = du.parse_pdb_feats('folded_sample', esmf_sample_path)
            sample_seq = du.aatype_to_seq(sample_feats['aatype'])

            # Calculate scTM of ESMFold outputs with reference protein
            _, tm_score = metrics.calc_tm_score(
                sample_feats['bb_positions'], esmf_feats['bb_positions'],
                sample_seq, sample_seq)
            rmsd = metrics.calc_aligned_rmsd(
                sample_feats['bb_positions'], esmf_feats['bb_positions'])
            if motif_mask is not None:
                sample_motif = sample_feats['bb_positions'][motif_mask]
                of_motif = esmf_feats['bb_positions'][motif_mask]
                motif_rmsd = metrics.calc_aligned_rmsd(
                    sample_motif, of_motif)
                mpnn_results['motif_rmsd'].append(motif_rmsd)
            mpnn_results['rmsd'].append(rmsd)
            mpnn_results['tm_score'].append(tm_score)
            mpnn_results['sample_path'].append(esmf_sample_path)
            mpnn_results['header'].append(header)
            mpnn_results['sequence'].append(string)

        # Save results to CSV
        csv_path = os.path.join(decoy_pdb_dir, 'sc_results.csv')
        mpnn_results = pd.DataFrame(mpnn_results)
        mpnn_results.to_csv(csv_path)
        return mpnn_results

    def run_folding(self, sequence, save_path):
        """Run ESMFold on sequence."""
        with torch.no_grad():
            output = self._folding_model.infer_pdb(sequence)

        with open(save_path, "w") as f:
            f.write(output)
        return output


@hydra.main(version_base=None, config_path="./sc_config/", config_name="inference")
def run(conf: DictConfig) -> None:

    # Read model checkpoint.
    print('Starting inference')
    start_time = time.time()
    evaluator = SelfConsistency(conf)
    evaluator.main()
    elapsed_time = time.time() - start_time
    print(f'Finished in {elapsed_time:.2f}s')

if __name__ == '__main__':
    run()
