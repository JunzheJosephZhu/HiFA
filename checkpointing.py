import torch
from torch.autograd import Variable
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler
import torch.utils.checkpoint
latents = Variable(torch.randn((2, 4, 64, 64), dtype=torch.float32, device=torch.device("cuda")), requires_grad=True)
unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="unet").to(torch.device("cuda")).requires_grad_()
t = torch.tensor(212, device=torch.device("cuda"), dtype=torch.float32, requires_grad=True)
text_embeddings = torch.randn((2, 77, 1024), device=torch.device("cuda"), dtype=torch.float32, requires_grad=True)
# unet.enable_gradient_checkpointing()
print(unet.is_gradient_checkpointing)
noise_pred = torch.utils.checkpoint.checkpoint(unet, latents, t, text_embeddings).sample
# noise_pred = unet(latents, t, text_embeddings).sample
print(noise_pred.requires_grad)