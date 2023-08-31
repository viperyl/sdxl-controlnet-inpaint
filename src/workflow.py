from diffusers import ControlNetModel, AutoencoderKL, EulerAncestralDiscreteScheduler
from diffusers import AutoPipelineForImage2Image, AutoPipelineForInpainting
from diffusers.utils import load_image
from PIL import Image
import numpy as np
import torch
import time
import json
import cv2
import os
from .pipeline import StableDiffusionXLInpaintControlNetPipeline


os.environ["http_proxy"] = "127.0.0.1:15777"
os.environ["https_proxy"] = "127.0.0.1:15777"

class sdxl_inpaint_controlnet_refiner:
    def __init__(self):
        controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-sdxl-1.0",
        torch_dtype=torch.float16
        )
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        self.pipe0 = StableDiffusionXLInpaintControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            torch_dtype=torch.float16,
            vae=vae
        )
        self.pipe0.enable_model_cpu_offload()
        
        self.pipe1 = AutoPipelineForInpainting.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
        self.pipe1.enable_model_cpu_offload()
        
        self.pipe2 = AutoPipelineForImage2Image.from_pipe(self.pipe1)
    
    
    def __call__(self, image, mask, c_image, prompt, negative_prompt, generator, controlnet_conditioning_scale,
                 guess_mode, guidance_scale, strength, control_guidance_start, control_guidance_end, guidance_rescale,
                 crops_coords_top_left, aesthetic_score, negative_aesthetic_score, eta):
        images = self.pipe0(
            prompt, 
            negative_prompt=negative_prompt,
            image=image, 
            mask_image=mask, 
            control_image=c_image,
            num_inference_steps=50,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            guess_mode = guess_mode,
            guidance_scale = guidance_scale,
            strength = strength,
            control_guidance_start = control_guidance_start,
            control_guidance_end = control_guidance_end,
            guidance_rescale = guidance_rescale,
            crops_coords_top_left = crops_coords_top_left,
            aesthetic_score = aesthetic_score,
            negative_aesthetic_score = negative_aesthetic_score,
            eta = eta,
            generator=generator,
        ).images
        
        image = self.pipe1(
            prompt=prompt,
            image=images[0],
            mask_image=mask,
            guidance_scale=8.0,
            num_inference_steps=100,
            strength=0.2,
            generator=generator,
            output_type="latent",
        ).images[0]
        
        image = self.pipe2(
            prompt=prompt,
            image=image,
            guidance_scale=8.0,
            num_inference_steps=100,
            strength=0.2,
            generator=generator,
        ).images[0]
        
        return image
        

    
