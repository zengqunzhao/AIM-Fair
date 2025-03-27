import argparse, os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from prompts import *
torch.set_grad_enabled(False)


def load_model_from_config(config, ckpt, device=torch.device("cuda"), verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, nargs="?", default="Portrait face photo of a smiling Caucasian male with blond hair.", help="the prompt to render")
    parser.add_argument("--outdir", type=str, nargs="?", help="dir to write results to")
    parser.add_argument("--steps", type=int, default=50, help="number of ddim sampling steps")
    parser.add_argument("--plms", action='store_true', help="use plms sampling")
    parser.add_argument("--dpm", action='store_true', help="use DPM (2) sampler")
    parser.add_argument("--fixed_code", action='store_true', help="if enabled, uses the same starting code across all samples")
    parser.add_argument("--ddim_eta", type=float, default=0.0, help="ddim eta (eta=0.0 corresponds to deterministic sampling")
    parser.add_argument("--H", type=int, default=512, help="image height, in pixel space")
    parser.add_argument("--W", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--C", type=int, default=4, help="latent channels")
    parser.add_argument("--f", type=int, default=8, help="downsampling factor, most often 8 or 16")
    parser.add_argument("--n_samples", type=int, default=15, help="how many samples to produce for each given prompt. A.k.a batch size")
    parser.add_argument("--n_iter", type=int, default=10, help="sample this often")
    parser.add_argument("--scale", type=float, default=9.0, help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))")
    parser.add_argument("--config", type=str, default="configs/stable-diffusion/v2-inference.yaml", help="path to config which constructs model")
    parser.add_argument("--ckpt", type=str, default="/data/scratch/acw717/v2-1_512-ema-pruned.ckpt", help="path to checkpoint of model")
    parser.add_argument("--seed", type=int, default=42, help="the seed (for reproducible sampling)")
    parser.add_argument("--precision", type=str, help="evaluate at this precision", choices=["full", "autocast"], default="autocast")
    parser.add_argument("--repeat", type=int, default=1, help="repeat each prompt in file this often")
    parser.add_argument("--device", type=str, help="Device on which Stable Diffusion will be run", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--target-sensitive", type=str, help="Smiling_Male, Smiling_Young")
    parser.add_argument("--prompt-type", type=str, help="Single_Prompt, Multiply_Prompt")
    opt = parser.parse_args()
    return opt


def main(opt):

    seed_everything(opt.seed)
    config = OmegaConf.load(f"{opt.config}")
    device = torch.device("cuda") if opt.device == "cuda" else torch.device("cpu")
    model = load_model_from_config(config, f"{opt.ckpt}", device)

    if opt.plms:
        sampler = PLMSSampler(model, device=device)
    elif opt.dpm:
        sampler = DPMSolverSampler(model, device=device)
    else:
        sampler = DDIMSampler(model, device=device)

    prompt_list = all_prompt_list[f"{opt.target_sensitive}_{opt.prompt_type}"]

    for prompts_title in tqdm(prompt_list, desc="Processing Groups"):

        prompts = prompt_list[prompts_title]
        out_path = f"{opt.outdir}/{opt.target_sensitive}/{opt.prompt_type}"
        os.makedirs(out_path, exist_ok=True)

        for prompt in prompts:
            batch_size = opt.n_samples
            data = [batch_size * [prompt]]

            sample_path = os.path.join(out_path, prompts_title)
            os.makedirs(sample_path, exist_ok=True)
            sample_count = 0
            base_count = len(os.listdir(sample_path))

            start_code = None
            if opt.fixed_code:
                start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

            precision_scope = autocast if opt.precision=="autocast" or opt.bf16 else nullcontext

            with torch.no_grad(), precision_scope(opt.device), model.ema_scope():
                    for n in range(opt.n_iter):
                        for prompts in data:
                            uc = None
                            if opt.scale != 1.0:
                                uc = model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = model.get_learned_conditioning(prompts)
                            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                            samples, _ = sampler.sample(S=opt.steps,
                                                        conditioning=c,
                                                        batch_size=opt.n_samples,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=opt.scale,
                                                        unconditional_conditioning=uc,
                                                        eta=opt.ddim_eta,
                                                        x_T=start_code)

                            x_samples = model.decode_first_stage(samples)
                            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                            for x_sample in x_samples:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                img = img.resize((224, 224), Image.Resampling.LANCZOS)
                                img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                                base_count += 1
                                sample_count += 1

if __name__ == "__main__":
    opt = parse_args()
    print('##### Generation Setting #####')
    for k, v in vars(opt).items():
        print(k,'=',v)
    print('############################')
    main(opt)
