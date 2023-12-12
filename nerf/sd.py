import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer, logging, DPTForDepthEstimation, DPTFeatureExtractor
from diffusers import PNDMScheduler, DDIMScheduler, DPMSolverMultistepScheduler, DiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from typing import Optional, Tuple, Union
from diffusers.configuration_utils import ConfigMixin, register_to_config
import matplotlib.pyplot as plt
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionDepth2ImgPipeline
from torchmetrics import PearsonCorrCoef
from .perpneg_utils import weighted_perpendicular_aggregator
import warnings

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import time
from torch.cuda.amp import custom_bwd, custom_fwd 
from typing import Optional, Union, Mapping
EPS = 1e-8



def add_tokens_to_model(learned_embeds: Mapping[str, Tensor], text_encoder: CLIPTextModel, 
        tokenizer: CLIPTokenizer, override_token: Optional[Union[str, dict]] = None) -> None:
    r"""Adds tokens to the tokenizer and text encoder of a model."""
    
    # Loop over learned embeddings
    new_tokens = []
    for token, embedding in learned_embeds.items():
        embedding = embedding.to(text_encoder.get_input_embeddings().weight.dtype)
        if override_token is not None:
            token = override_token if isinstance(override_token, str) else override_token[token]
        
        # Add the token to the tokenizer
        num_added_tokens = tokenizer.add_tokens(token)
        if num_added_tokens == 0:
            raise ValueError((f"The tokenizer already contains the token {token}. Please pass a "
                               "different `token` that is not already in the tokenizer."))
  
        # Resize the token embeddings
        text_encoder.resize_token_embeddings(len(tokenizer))
  
        # Get the id for the token and assign the embeds
        token_id = tokenizer.convert_tokens_to_ids(token)
        text_encoder.get_input_embeddings().weight.data[token_id] = embedding    
        new_tokens.append(token)

    print(f'Added {len(new_tokens)} tokens to tokenizer and text embedding: {new_tokens}')


def add_tokens_to_model_from_path(learned_embeds_path: str, text_encoder: CLIPTextModel, 
        tokenizer: CLIPTokenizer, override_token: Optional[Union[str, dict]] = None) -> None:
    r"""Loads tokens from a file and adds them to the tokenizer and text encoder of a model."""
    learned_embeds: Mapping[str, Tensor] = torch.load(learned_embeds_path, map_location='cpu')
    add_tokens_to_model(learned_embeds, text_encoder, tokenizer, override_token)


    
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

class StableDiffusion(nn.Module):
    def __init__(self, device, loss_type, sd_version='2.1', hf_key=None, no_sds=False, opt=None):
        super().__init__()

        self.opt=opt
        self.device = device
        self.loss_type = loss_type
        self.sd_version = sd_version
        self.no_sds = no_sds
        self.fp16 = opt.fp16

        self.name = 'sd'
        
        print('self.opt.h, self.opt.w', self.opt.h, self.opt.w)
        self.original_size = (self.opt.h, self.opt.w)
        
        print(f'[INFO] loading stable diffusion...')
        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
            # model_key = "/home/joseph/.cache/huggingface/diffusers/models--stabilityai--stable-diffusion-2-1-base/snapshots/88bb1a46821197d1ac0cb54d1d09fb6e70b171bc" 
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '5.1':
            model_key = 'SG161222/Realistic_Vision_V5.1_noVAE'
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
            print('model_key', model_key)
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        # get depth model if using depth predictor supservision
        if self.opt.lambda_depthreg > 0:
            # i think the weight here is different from Intel/dpt-large. That one causes NaN values for some reason.
            depthpipe = StableDiffusionDepth2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-depth").to(self.device)
            self.feature_extractor = depthpipe.feature_extractor
            self.depth_estimator = depthpipe.depth_estimator
            del depthpipe.vae
            del depthpipe.unet
            del depthpipe.text_encoder
            torch.cuda.empty_cache()

            # self.feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
            # self.depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(self.device)

        pipe = DiffusionPipeline.from_pretrained(model_key, use_safetensors=True).to(self.device)
        # pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
        
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        

        if opt.learned_embeds_path is not None:  # add textual inversion tokens to model
            add_tokens_to_model_from_path(
                opt.learned_embeds_path, self.text_encoder, self.tokenizer
            )

        self.pipe = pipe
        if type(pipe) is StableDiffusionXLPipeline:
            # self.add_time_ids = pipe._get_add_time_ids(
            #     (self.opt.interp_size, self.opt.interp_size), (0, 0), (self.opt.interp_size,  self.opt.interp_size), dtype=torch.float16
            # ).to(self.device)
            self.add_time_ids = pipe._get_add_time_ids(
                self.original_size, (0, 0), (self.opt.interp_size,  self.opt.interp_size), dtype=torch.float16
            ).to(self.device)
            self.vae.to(dtype=torch.float32)
            self.is_xl = True
        elif type(pipe) is  StableDiffusionXLImg2ImgPipeline:            
            self.add_time_ids, _ = pipe._get_add_time_ids(
                (self.opt.interp_size, self.opt.interp_size), (0, 0), (self.opt.interp_size,  self.opt.interp_size), 6.0, 2.5, dtype=torch.float16
            ).to(self.device)
            self.vae.to(dtype=torch.float32)
            self.is_xl = True
        else:
            self.is_xl = False
        self.size_scale = 1 / (self.opt.interp_size * self.opt.interp_size) * (self.opt.w * self.opt.h)

        if is_xformers_available():
            self.unet.enable_xformers_memory_efficient_attention()
        
        if self.opt.scheduler == "ddim":
            self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        elif self.opt.scheduler == "solver":
            self.scheduler = DPMSolverMultistepScheduler.from_pretrained(model_key, subfolder="scheduler", solver_order=3)
        # self.scheduler = PNDMScheduler.from_pretrained(model_key, subfolder="scheduler")

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * self.opt.min_percent)
        self.max_step = int(self.num_train_timesteps * self.opt.max_percent)
        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        if self.opt.use_pearson:
            self.pearson = PearsonCorrCoef().to(self.device)

        print(f'[INFO] loaded stable diffusion!')

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        # prompt[str]
        if self.is_xl:
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt=prompt,
                device=self.device,
                num_images_per_prompt=1,
                negative_prompt=[""]
            )
            
            return prompt_embeds, pooled_prompt_embeds
        else:
            inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
            embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
            return embeddings, torch.zeros_like(embeddings)


    def encode_latent(self, img, requires_grad=False):
        with torch.set_grad_enabled(requires_grad):
            with torch.cuda.amp.autocast(enabled=self.fp16):
                    latents = self.encode_imgs(img)
        return latents                                   
                
    def train_step(self, text_embeddings, text_embeddings2, neg_prompt_weights, pred_rgb, outputs, guidance_scale=100, global_step=0, is_normal=False):
        B = pred_rgb.shape[0]
        K = (text_embeddings.shape[0] // B) - 1 # maximum number of prompts      
        
        if self.is_xl and self.fp16:
            # deprecated if fp32
            self.vae.post_quant_conv.to(text_embeddings.dtype)
            self.vae.decoder.conv_in.to(text_embeddings.dtype)
            self.vae.decoder.mid_block.to(text_embeddings.dtype)

        if is_normal and not self.opt.encode_normal:
            latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (self.opt.interp_size, self.opt.interp_size), mode='bilinear', align_corners=False)

            # encode image into latents with vae, requires grad!
            # with torch.cuda.amp.autocast(enabled=not self.is_xl):
            with torch.cuda.amp.autocast(enabled=self.fp16):
                latents = self.encode_imgs(pred_rgb_512)
        
        if self.opt.timestep_scheduling == 'sqrt':
            percentage = np.sqrt(float(global_step) / self.opt.iters) # dont forget to change sigma schedule with this
            t = (self.max_step - self.min_step) * (1 - percentage) + self.min_step 
            t = torch.tensor(t, device=self.device).to(int)
        elif self.opt.timestep_scheduling == 'linear':
            percentage = float(global_step) / self.opt.iters
            t = (self.max_step - self.min_step) * (1 - percentage) + self.min_step 
            t = torch.tensor(t, device=self.device).to(int)
        elif self.opt.timestep_scheduling == 'random':
            percentage = torch.rand((1,), device=self.device)[0]
            t = (20 + 960 * percentage).to(int)
        elif self.opt.timestep_scheduling == 'cosine':
            percentage = np.cos(np.pi - float(global_step) / self.opt.iters * np.pi) / 2 + 0.5
            t = (self.max_step - self.min_step) * (1 - percentage) + self.min_step 
            t = torch.tensor(t, device=self.device).to(int)

        if self.opt.sds_weight == "constant":
            w = 0.05 # default scale coefficient for reasonable weights compared to other losses
        elif self.opt.sds_weight == "noisematch":
            w = 0.05 * self.alphas_cumprod[t] ** 0.5 / (1 - self.alphas_cumprod[t]) ** 0.5
        elif self.opt.sds_weight == "sdslike":
            w = 0.05 * self.alphas_cumprod[t] ** 0.5 * (1 - self.alphas_cumprod[t]) ** 0.5
        elif self.opt.sds_weight == "strongerlower":
            w = 0.05 * self.alphas_cumprod[t] ** 1.0 / (1 - self.alphas_cumprod[t]) ** 0.5

        if self.opt.scheduler == "solver":
            self.scheduler.set_timesteps(self.opt.denoising_steps)
            timestep_flipped = self.scheduler.timesteps.flip(-1).to(t.device)
            timestep = timestep_flipped[torch.searchsorted(timestep_flipped, t)]
        elif self.opt.scheduler == "ddim":
            timestep = t
        else:
            raise NotImplementedError
        
        with torch.no_grad():
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, timestep)
            # TODO: for DPM solver, should add noise wrt first timestep in scheduler timesteps
            latent_model_input = torch.cat([latents_noisy] * (1 + K))
            if self.opt.denoising_steps == 1: # do single step denoising
                if self.is_xl:     
                    added_cond_kwargs = {"text_embeds": text_embeddings2, "time_ids": self.add_time_ids.repeat(latent_model_input.shape[0], 1)}
                    unet_output = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings,
                        cross_attention_kwargs=None,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]
                else:
                    unet_output = self.unet(latent_model_input, timestep, encoder_hidden_states=text_embeddings).sample
                # perform guidance (high scale from paper!)
                noise_pred_uncond, noise_pred_text = unet_output[:B], unet_output[B:]
                delta_noise_preds = noise_pred_text - noise_pred_uncond.repeat(K, 1, 1, 1)
                noise_pred = noise_pred_uncond + guidance_scale * weighted_perpendicular_aggregator(delta_noise_preds, neg_prompt_weights, B)            
                latents_denoised = (latents_noisy - (1 - self.alphas_cumprod[timestep]) ** 0.5 * noise_pred) / self.alphas_cumprod[timestep] ** 0.5
            else: # do multistep denoising
                latents_denoised = self.denoise(B, K, latents_noisy, 
                                                text_embeddings, text_embeddings2, 
                                                neg_prompt_weights, 
                                                timestep, 
                                                self.opt.denoising_steps, 
                                                guidance_scale=guidance_scale, eta=self.opt.eta)

        error = latents - latents_denoised.detach()
        if self.opt.clip_grad:
            thres = 10 / 0.05 * (1 - self.alphas_cumprod[timestep]) ** 0.5 # threshold scales with noise strength. Want to clip grad at 10 at max timestep
            error_clamped = error.clamp(-thres, thres)
            if not torch.allclose(error, error_clamped):
                warnings.warn("grad clip just happened. Might be a gradient explosion")
            error = error_clamped
        loss_recon = torch.sum(error ** 2).item() # scalar value for return only
        lambda_sds = self.opt.lambda_sds
        loss = lambda_sds * torch.sum(error ** 2) / 2 * w * self.size_scale

        if not is_normal:
            if self.opt.lambda_rgb > 0:
                # compute rgb pseudo ground truth
                with torch.no_grad():                    
                    # with torch.cuda.amp.autocast(enabled=not self.is_xl):
                    with torch.cuda.amp.autocast(enabled=self.fp16):
                        rgb_denoised = self.decode_latents(latents_denoised.to(text_embeddings.dtype))
                        if torch.isnan(rgb_denoised).any():
                            print('L294 latents_denoised', latents_denoised.dtype, latents_denoised.max(), latents_denoised.min(), 'latents', latents.dtype, latents.max(), latents.min(), 'rgb_denoised', rgb_denoised.dtype, rgb_denoised.max(), rgb_denoised.min())
                        assert not torch.isnan(rgb_denoised).any()
                        
                if self.opt.color_bias == 'grayscale':
                    color_coeff = torch.Tensor([0.299, 0.587, 0.114]).reshape(1, 3, 1, 1).to(self.device) * 3
                elif self.opt.color_bias == '':
                    color_coeff = torch.Tensor([1, 1, 1]).reshape(1, 3, 1, 1).to(self.device)
                else:
                    raise NotImplementedError
                
                if self.opt.intensity_only:
                    error_rgb = (pred_rgb_512 * color_coeff).sum(1, keepdim=True) - (rgb_denoised * color_coeff).sum(1, keepdim=True)
                else:
                    error_rgb = pred_rgb_512 * color_coeff - rgb_denoised * color_coeff
                # add rgb recon loss
                loss +=  self.opt.lambda_rgb * torch.sum(error_rgb ** 2) / 2 * w * self.size_scale
            
            
            if self.opt.lambda_depthreg > 0:
                weights = outputs["weights"]
                weights_sum = weights.sum(dim=-1).reshape(pred_rgb.shape[0], pred_rgb.shape[2], pred_rgb.shape[3]) # [N, h, w]
                depth = outputs["depth"].reshape(pred_rgb.shape[0], pred_rgb.shape[2], pred_rgb.shape[3]) # [N, 512, 512]
    
                with torch.no_grad():
                    mask = weights_sum > 0.5
                    if self.opt.depthreg_schedule == "anneal":
                        lambda_depthreg = self.opt.lambda_depthreg * (1 - percentage)
                    elif self.opt.depthreg_schedule == "constant":
                        lambda_depthreg = self.opt.lambda_depthreg
                    elif self.opt.depthreg_schedule == "quadratic":
                        lambda_depthreg = self.opt.lambda_depthreg * percentage * (1 - percentage) * 4
                    pixel_values = self.feature_extractor(images=(pred_rgb_512 * 255).to(torch.uint8), return_tensors="pt").pixel_values
                    pixel_values = pixel_values.to(pred_rgb_512.device)
                    # this is actually disparity values
                    disparity_pred = self.depth_estimator(pixel_values).predicted_depth
                    disparity_pred = torch.nn.functional.interpolate(
                        disparity_pred.unsqueeze(1),
                        size=(depth.shape[1], depth.shape[2]),
                        mode="bicubic",
                        align_corners=False,
                    ).to(depth.dtype)[:, 0]
                    # optimal estimate target
                    target = disparity_pred[mask]
                    target_normalized = (target - target.mean()) / target.std()

                if self.opt.use_pearson:
                    loss += lambda_depthreg * (1 - self.pearson(depth[mask] / weights_sum[mask], 1 / target))
                else:
                    disparity = 1 / (depth[mask] / weights_sum[mask])
                    pred = disparity - disparity.mean()
                    pred_rescaled = torch.sum(pred * target_normalized) / torch.sum(pred ** 2) * pred
                    error = pred_rescaled - target_normalized
                    loss += lambda_depthreg * torch.mean(error ** 2) / 2 * w * (256 * 256) 
                    # for visualization
                    # target_map = torch.zeros_like(depth_map)
                    # target_map[mask] = target_normalized
                    # pred_map = torch.zeros_like(depth)
                    # pred_map[mask] = pred_rescaled
        return loss, loss_recon

    def denoise(self, B, K, latents, text_embeddings, text_embeddings2, weights, timestep, num_inference_steps, guidance_scale, eta):
        '''
        Do multistep ddim/DPMSolver
        '''
        self.scheduler.set_timesteps(num_inference_steps)
        with torch.no_grad():
            if self.opt.scheduler == "ddim":
                step_ratio = self.scheduler.config.num_train_timesteps // num_inference_steps
                schedule = range(timestep, -1, -step_ratio)
            elif self.opt.scheduler == "solver":
                schedule_flipped = self.scheduler.timesteps.to(timestep.device).flip(-1)
                schedule = schedule_flipped[:torch.searchsorted(schedule_flipped, timestep) + 1].flip(-1)
                            
            for t in schedule:
                latent_model_input = torch.cat([latents] * (1 + K))
                if self.is_xl:
                    added_cond_kwargs = {"text_embeds": text_embeddings2, "time_ids": self.add_time_ids.repeat(latent_model_input.shape[0], 1)}
                    unet_output = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings,
                        cross_attention_kwargs=None,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]
                else:
                    unet_output = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                # perform guidance (high scale from paper!)
                noise_pred_uncond, noise_pred_text = unet_output[:B], unet_output[B:]
                delta_noise_preds = noise_pred_text - noise_pred_uncond.repeat(K, 1, 1, 1)
                noise_pred = noise_pred_uncond + guidance_scale * weighted_perpendicular_aggregator(delta_noise_preds, weights, B)            

                # compute the previous noisy sample x_t -> x_t-1
                if self.opt.scheduler == 'ddim':
                    latents = self.scheduler.step(noise_pred, t, latents, eta=eta)['prev_sample']
                elif self.opt.scheduler == "solver":
                    latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']    
                else:
                    raise NotImplementedError 
        return latents

    def decode_latents(self, latents):

        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        
        return imgs
    

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents





