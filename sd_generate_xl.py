from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, DPMSolverMultistepScheduler
from diffusers import DiffusionPipeline
import torch
import huggingface_hub
import matplotlib.pyplot as plt

huggingface_hub.login("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")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-0.9", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
# pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-0.9", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")

pipe.to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, solver_order=2)

use_refiner = False
refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-0.9", torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
)
refiner.to("cuda")

prompt = "a stack of pancakes covered in maple syrup"

with torch.no_grad():
    image = pipe(prompt=prompt, output_type="latent" if use_refiner else "pil", num_inference_steps=20, height=1024, width=1024).images[0]
    # image = refiner(prompt=prompt, image=image[None, :], width=512, height=512, denoising_start=0.7).images[0]
    plt.imshow(image)
    plt.show()