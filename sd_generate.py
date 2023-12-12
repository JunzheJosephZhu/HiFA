
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionDepth2ImgPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler, DPMSolverMultistepScheduler
import matplotlib.pyplot as plt
import os
# pipe = StableDiffusionDepth2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-depth", torch_dtype=torch.float16).to("cuda")

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16)
pipe = pipe.to("cuda")
scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, solver_order=2)
pipe.scheduler = scheduler

text = "head of thanos"
# text = "an alien blaster pistol from fallout 4"
# text = "a glass pendant with eye of sauron inside"
# text = "iron throne from game of thrones"
# text = "a pyre made from bone, no fire"
# text = "game asset of a leather shoe made from dragon skin"
# text = "full-body photo of burning man statue in greek armor made of scaffold, arms hailing"
# text = "a modern style ebony statue of an african tribal woman"
# text = "a mechanical octopus with tiled metal skin"
# text = "matte painting of a castle made of cheesecake surrounded by a moat made of ice cream"
# text = "a hamburger"
# os.makedirs(f"nerf/graf/data/{text}", exist_ok=True)
for d in ['', 'front', 'side', 'back', 'overhead', 'bottom']:
    # construct dir-encoded text
    if d:
        prompt = f"{text}, {d} view"
    else:
        prompt = text
    for i in range(50):
        with torch.no_grad():
            image = pipe(prompt, guidance_scale=7.5, num_inference_steps=12, eta=1).images[0]
            plt.imshow(image)
            plt.show()
            # image.save(f"nerf/graf/data/{text}/{d}_{i}.png")

# os.makedirs(f"nerf/graf/latents/{text}", exist_ok=True)
# for d in ['', 'front', 'side', 'back', 'overhead', 'bottom']:
#     # construct dir-encoded text
#     if d:
#         prompt = f"{text}, {d} view"
#     else:
#         prompt = text
#     for i in range(50):
#         with torch.no_grad():
#             latents = pipe(prompt, output_type='latent').images[0]
#             torch.save(latents, f"nerf/graf/latents/{text}/{d}_{i}.pt")
