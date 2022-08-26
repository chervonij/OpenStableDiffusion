import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from interact import interact as io

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from pathlib import Path
import pickle
import time

import accelerate
import k_diffusion as K
import torch.nn as nn

from transformers import logging
logging.set_verbosity(logging.ERROR)

def load_saved_options(path):
    path = Path(str(path.parent)+'/config.dat')
    if path.exists():
        data = pickle.loads(path.read_bytes())
    else:
        data = {}

    return data

def save_options(path, data):
    path = Path(str(path.parent)+'/config.dat')
    path.write_bytes(pickle.dumps(data))

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())
def load_model_from_config(config, ckpt, verbose=False):
    #print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    #if "global_step" in pl_sd:
        #print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="./configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )

    opt = parser.parse_args()

    options = load_saved_options(Path(opt.ckpt))


    options['prompt'] = io.input_str("Prompt", options.get('prompt', 'Portrait'), help_message="Prompt to render")

    options['batch_size'] = io.input_int("Batch size", options.get('batch_size', 4), help_message="How much images generated per iteration")
    options['n_iter'] = io.input_int("Iterations", options.get('n_iter', 5), help_message="Number of iterations")

    options['scale'] = io.input_number("Scale", options.get('scale', 7.5), add_info="0.0..30.0", help_message="Unconditional guidance scale")
    options['ddim_steps'] = io.input_int("Diffusion steps", options.get('ddim_steps', 80), help_message="Number of sampling steps. More - better quality.")

    options['sampler_type'] = io.input_str("Sampler type", options.get('sampler_type', 'klms'), ['klms','plms','ddim'], help_message="Method of images sampling from random")
    options['ddim_eta'] = 0.0 if options['sampler_type'] != 'ddim' else io.input_number("DDIM Eta", options.get('ddim_eta', 0.5), add_info="0.0..1.0", help_message="Unknown")

    options['H'] = io.input_int("Height", options.get('H', 512), help_message="Height of generated images")
    options['W'] = io.input_int("Width", options.get('W', 512), help_message="Width of generated images")

    options['save_grid'] = io.input_bool("Save grid", options.get('save_grid', False), help_message="Save all generated images also as grid")
    options['n_rows'] = 1 if not options['save_grid'] else io.input_int("Rows in grid", options.get('n_rows', options['batch_size'].copy()), help_message="Rows in grid")

    options['seed'] = io.input_int ("Random seed", options.get('seed', -1), help_message="Prompt to render. Keep default to get current time as seed")

    save_options(Path(opt.ckpt), options)

    print('\n')

    C, f = 4, 8
    precision = 'autocast'

    prompt, batch_size, n_iter, scale, ddim_steps = options['prompt'], options['batch_size'], options['n_iter'], options['scale'], options['ddim_steps']
    sampler_type, ddim_eta, H, W = options['sampler_type'], options['ddim_eta'], options['H'], options['W']
    save_grid, n_rows, seed = options['save_grid'], options['n_rows'], options['seed']

    #seed_everything(opt.seed)
    if seed == -1:
        seed_everything(time.time())
    else:
        seed_everything(seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    accelerator = accelerate.Accelerator()
    if sampler_type == 'klms':
        model_wrap = K.external.CompVisDenoiser(model)
        device = accelerator.device
        class CFGDenoiser(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.inner_model = model

            def forward(self, x, sigma, uncond, cond, cond_scale):
                x_in = torch.cat([x] * 2)
                sigma_in = torch.cat([sigma] * 2)
                cond_in = torch.cat([uncond, cond])
                uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
                return uncond + (cond - uncond) * cond_scale

    elif sampler_type == 'plms':
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    assert prompt is not None
    data = [batch_size * [prompt]]

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(n_iter, desc="Sampling", disable =not accelerator.is_main_process):
                    uc = None
                    prompts = data[0]
                    if scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = model.get_learned_conditioning(prompts)
                    shape = [C, H//f, W//f]
                    if not sampler_type == 'klms':
                        samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                        conditioning=c,
                                                        batch_size=batch_size,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=uc,
                                                        eta=ddim_eta,
                                                        x_T=start_code)
                    else:
                        sigmas = model_wrap.get_sigmas(ddim_steps)
                        if start_code is not None:
                            x = start_code
                        else:
                            x = torch.randn([batch_size, *shape], device=device) * sigmas[0] # for GPU draw
                        model_wrap_cfg = CFGDenoiser(model_wrap)
                        extra_args = {'cond': c, 'uncond': uc, 'cond_scale': scale}
                        samples_ddim = K.sampling.sample_lms(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=not accelerator.is_main_process)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                    if sampler_type == 'klms':
                        x_samples_ddim = accelerator.gather(x_samples_ddim)

                    for x_sample in x_samples_ddim:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        img = Image.fromarray(x_sample.astype(np.uint8))
                        img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                        base_count += 1

                    if save_grid:
                        all_samples.append(x_samples_ddim)

                if save_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                    grid_count += 1

                toc = time.time()

    print('Done!')


if __name__ == "__main__":
    main()
