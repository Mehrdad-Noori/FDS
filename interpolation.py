import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler



def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
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


def get_random_floats(num, lower, upper, dec=2):
    random_floats = [round(i, dec) for i in torch.FloatTensor(num).uniform_(lower, upper).tolist()]
    return torch.tensor(random_floats)



def main():
    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--training_domains",
        type=str,
        nargs="+",
        help="the training domains"
    )

    parser.add_argument(
        "--augment_domains",
        type=str,
        nargs="+",
        help="Two domains that should be interpolated"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
    )


    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--fixed_code", 
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )

    parser.add_argument(
        "--int_bounds",
        type=float,
        nargs="+",
        help="the lower and upper bound  for interpolation weights",
    )

    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        help="the target classes",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--iter_per_class",
        type=int,
        default=1,
        help="sample this often for each class",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=256,
        help="image height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=256,
        help="image width, in pixel space",
    )

    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f", 
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=8,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )

    parser.add_argument(
        "--scale_bounds",
        type=float,
        nargs="+",
        help="unconditional guidance scale lower and upper bound: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--dyn", 
        type=float,
        help="dynamic thresholding from Imagen, in latent space (TODO: try in pixel space with intermediate decode)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

    opt = parser.parse_args()
    seed_everything(opt.seed)


    all_classes = opt.classes
    int_bounds = opt.int_bounds
    assert len(int_bounds)==2 and int_bounds[0] <= int_bounds[1]

    scale_bounds = opt.scale_bounds
    assert len(scale_bounds)==2 and scale_bounds[0] <= scale_bounds[1]

    training_domains = opt.training_domains 
    augment_domains = opt.augment_domains

    print(f"training_domains: {training_domains}")
    print(f"augment_domains: {augment_domains}")

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    
    sampler = DDIMSampler(model)
    batch_size = opt.n_samples

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
  

    new_domain_name = augment_domains[0] + '_' + augment_domains[1]
    print(f"+++ New Domain Name: {new_domain_name}")

    for c_idx, class_name in enumerate(all_classes):
        print(f"++++ creating the class {c_idx+1}/{len(all_classes)}: {class_name}")

        prompt1 = augment_domains[0] + f", {class_name}"
        prompt2 = augment_domains[1] + f", {class_name}"

  
        print(f"prompt1: {prompt1}")
        print(f"prompt2: {prompt2}")

        prompt1 = batch_size * [prompt1]
        prompt2 = batch_size * [prompt2]

        ### configure path
        outpath = os.path.join(opt.outdir, new_domain_name)
        class_path = os.path.join(outpath, class_name)
        os.makedirs(class_path, exist_ok=True) 
        base_count = len(os.listdir(class_path))

        precision_scope = autocast if opt.precision=="autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    tic = time.time()
                    for n in trange(opt.iter_per_class, desc="Sampling"):
                        
                        # Get random interpolation weights
                        int_weights = get_random_floats(batch_size, int_bounds[0], int_bounds[1], dec=2) # shape: (B, )
                        int_weights = int_weights.to(device)
                        int_weights = int_weights.unsqueeze(-1).unsqueeze(-1)
                        
                        # Get random cfg scales
                        scales = get_random_floats(batch_size, scale_bounds[0], scale_bounds[1], dec=1) # shape: (B, )
                        scales = scales.to(device)
                   
                        uc = None

                        if not torch.all(scales == 1.):
                            uc = model.get_learned_conditioning(batch_size * [''])


                        c1 = model.get_learned_conditioning(prompt1)
                        c2 = model.get_learned_conditioning(prompt2)



                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]                        
                        samples_ddim, _ = sampler.sample_condition_interpolation(S=opt.ddim_steps,
                                                            conditioning1=c1,
                                                            conditioning2=c2,
                                                            batch_size=opt.n_samples,
                                                            int_weights=int_weights,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=scales,
                                                            unconditional_conditioning=uc,
                                                            eta=opt.ddim_eta,
                                                            dynamic_threshold=opt.dyn,
                                                            x_T=start_code,)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                        
                        for i, x_sample in enumerate(x_samples_ddim):
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            img_save_name = os.path.join(class_path, f"{base_count:05}_m2_s{opt.seed}_cfg{scales[i]:.3f}_i{int_weights.squeeze()[i]:.3f}.png")
                            Image.fromarray(x_sample.astype(np.uint8)).save(img_save_name)
                            base_count += 1



                    toc = time.time()

        print(f"Samples for the class '{class_name}' are ready. Please find them at: \n{outpath}\n"
                f"Sampling duration: {toc - tic:.2f} seconds. This results in a rate of "
                f"{opt.iter_per_class * opt.n_samples / (toc - tic):.2f} samples per second.")


if __name__ == "__main__":
    main()