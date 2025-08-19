#!/usr/bin/env python
# coding=utf-8
# Copyright (C) 2025 AIDC-AI
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at 
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License.

from typing import Optional, Union, List, Dict, Any

import math
import os
import torch
import torch.nn as nn
from diffusers import DiffusionPipeline, EulerDiscreteScheduler, SchedulerMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import logging
from PIL import Image

from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

def get_noise(
    num_samples: int,
    channel: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    return torch.randn(
        num_samples,
        channel,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )

class ChatsSD15Pipeline(DiffusionPipeline, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        unet_win: nn.Module,
        unet_lose: nn.Module,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        vae: AutoencoderKL,
        scheduler: SchedulerMixin
    ):
        super().__init__()

        self.register_modules(
            unet_win=unet_win,
            unet_lose=unet_lose,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            scheduler=scheduler
        )

    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        **kwargs,
    ) -> "ChatsSD15Pipeline":

        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    def save_pretrained(self, save_directory: Union[str, os.PathLike]):
        super().save_pretrained(save_directory)

    @torch.no_grad()
    def encode_text(self, tokenizer, text_encoder, prompt):

        with torch.no_grad():
            text_inputs = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")    
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(text_input_ids.to('cuda'))
            prompt_embeds = prompt_embeds[0]
        
        prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=text_encoder.device)

        return prompt_embeds

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        latents: torch.FloatTensor = None,
        height: int = 1024,
        width: int = 1024,
        seed: int = 0,
        alpha: float=0.5
    ):  
        if isinstance(prompt, str):
            prompt = [prompt]

        device = self.unet_win.device

        tokenizer = self.tokenizer
        text_encoder = self.text_encoder

        prompt_embeds = self.encode_text(tokenizer, text_encoder, prompt)
        negative_prompt_embeds = self.encode_text(tokenizer, text_encoder, "")

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        bs = len(prompt)
        channel = self.vae.config.latent_channels
        height = 16 * (height // 16)
        width = 16 * (width // 16)

        # prepare input
        latents = get_noise(
            bs,
            channel,
            height,
            width,
            device=device,
            dtype=self.unet_win.dtype,
            seed=seed,
        )
        latents = latents * self.scheduler.init_noise_sigma

        for i, t in enumerate(timesteps):
            latent_model_input = self.scheduler.scale_model_input(latents, t)

            pred_win = self.unet_win(latent_model_input, t, encoder_hidden_states=prompt_embeds, return_dict=False)[0]
            pred_lose = self.unet_lose(latent_model_input, t, encoder_hidden_states=prompt_embeds * (- alpha) + negative_prompt_embeds * (1. + alpha), return_dict=False)[0]

            noise_pred = pred_win + guidance_scale * (pred_win - pred_lose)
            latents = self.scheduler.step(noise_pred, t, latents, generator=None, return_dict=False)[0]

        x = latents.float()

        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float32):
                if hasattr(self.vae.config, 'scaling_factor') and self.vae.config.scaling_factor is not None:
                    x = x / self.vae.config.scaling_factor
                if hasattr(self.vae.config, 'shift_factor') and self.vae.config.shift_factor is not None:
                    x = x + self.vae.config.shift_factor
                x = self.vae.decode(x, return_dict=False)[0]

        # bring into PIL format and save
        x = (x / 2 + 0.5).clamp(0, 1)
        x = x.cpu().permute(0, 2, 3, 1).float().numpy()
        images = (x * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

        return pil_images