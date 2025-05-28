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

import argparse
import io
import logging
import math
import os
import random
import shutil
import sys
from pathlib import Path
import torch.nn as nn

import accelerate
import datasets
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from transformers.utils import ContextManagers
from diffusers.utils.torch_utils import is_compiled_module
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, StableDiffusionXLPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available

import functools
from torchvision.transforms.functional import crop
from transformers import AutoTokenizer
from datasets import load_dataset


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.20.0")

logger = get_logger(__name__, log_level="INFO")

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None,
                        # was random for submission, need to test that not distributing same noise etc across devices
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help=(
            "The resolution for input images, all the images in the dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=100
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=2000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-8,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_adafactor", action="store_true", help="Whether or not to use adafactor (should save mem)"
    )
    # Bram Note: Haven't looked @ this yet
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )

    ## CHATS
    parser.add_argument("--beta_dpo", type=float, default=1000, help="The beta DPO temperature controlling strength of KL penalty")
    parser.add_argument(
        "--puncond",
        type=float,
        default=0.5,
        help="Probability of prompts for lose model to be zero",
    )
    parser.add_argument(
        "--hard_skip_resume", action="store_true", help="Load weights etc. but don't iter through loader for loader resume, useful b/c resume takes forever"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--split", type=str, default='train', help="Datasplit"
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="CHATS training",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    ## SDXL
    print("Running CHATS-SDXL")
    if args.resolution is None:
      args.resolution = 1024
            
    return args

DATASET_NAME_MAPPING = {
    "data-is-better-together/open-image-preferences-v1-binarized": ("chosen", "rejected", "prompt"),
}


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt_sdxl(batch, text_encoders, tokenizers, proportion_empty_prompts, caption_column, is_train=True):
    prompt_embeds_list = []
    prompt_batch = batch[caption_column]

    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to('cuda'),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds}




def main():
    
    args = parse_args()
    
    #### START ACCELERATOR BOILERPLATE ###
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index) # added in + term, untested

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    ### END ACCELERATOR BOILERPLATE
    
    
    ### START DIFFUSION BOILERPLATE ###
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision, use_fast=False)
    tokenizer_two = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision, use_fast=False)


    # Not sure if we're hitting this at all
    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    
    # BRAM NOTE: We're not using deepspeed currently so not sure it'll work. Could be good to add though!
    # 
    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        # SDXL has two text encoders
        text_encoder_one = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision)
        text_encoder_two = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision)

        text_encoders = [text_encoder_one, text_encoder_two]
        tokenizers = [tokenizer_one, tokenizer_two]

        # Can custom-select VAE (used in original SDXL tuning)
        vae = AutoencoderKL.from_pretrained(args.pretrained_vae_model_name_or_path, subfolder=None, revision=args.revision)
        ref_unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision)

    
    unet_win = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision)
    unet_lose = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision)
    unet = nn.ModuleDict({"win" : unet_win, "lose" : unet_lose})


    # Freeze vae, text_encoder(s), reference unet
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    ref_unet.requires_grad_(False)
    
    print(sum([p.numel() for p in unet.parameters() if p.requires_grad]) / 1000000, 'M parameters')

    # xformers efficient attention
    if is_xformers_available():
        import xformers

        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            logger.warn(
                "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
            )
        unet["win"].enable_xformers_memory_efficient_attention()
        unet["lose"].enable_xformers_memory_efficient_attention()
    else:
        raise ValueError("xformers is not available. Make sure it is installed correctly")


    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:

            torch.save(unwrap_model(unet).state_dict(), os.path.join(output_dir, "unet_model_weights.pth"))

            for i, model in enumerate(models):

                if len(weights) > 0:
                    weights.pop()

    def load_model_hook(models, input_dir):

        if os.path.exists(os.path.join(input_dir, "unet_model_weights.pth")):
            state_dict = torch.load(os.path.join(input_dir, "unet_model_weights.pth"))
        else:
            raise NotImplementedError

        unwrap_model(unet).load_state_dict(state_dict)
        for _ in range(len(models)):
            model = models.pop()

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        print("Enabling gradient checkpointing")
        unet["win"].enable_gradient_checkpointing()
        unet["lose"].enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    params_to_optimize = [p for p in unet.parameters() if p.requires_grad]
    for n, p in unet.named_parameters():
        if p.requires_grad:
            print(n)

    if args.use_adafactor:
        optimizer = transformers.Adafactor(params_to_optimize,
                                           lr=args.learning_rate,
                                           weight_decay=args.adam_weight_decay,
                                           clip_threshold=1.0,
                                           scale_parameter=False,
                                          relative_step=False)
    else:
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    # load dataset
    dataset = load_dataset(args.dataset_name)    

    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)

    image_column_win = dataset_columns[0] 
    image_column_lose = dataset_columns[1]
    caption_column = dataset_columns[2]

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    
    #### START PREPROCESSING/COLLATION ####
    def preprocess_train(examples):
        all_pixel_values = []
        for col_name in [image_column_win, image_column_lose]:
            images = [Image.open(io.BytesIO(im_bytes['bytes'])).convert("RGB") for im_bytes in examples[col_name]]
            pixel_values = [train_transforms(image) for image in images]
            all_pixel_values.append(pixel_values)
        im_tup_iterator = zip(*all_pixel_values)
        combined_pixel_values = []
        for im_tup in im_tup_iterator:
            combined_im = torch.cat(im_tup, dim=0)
            combined_pixel_values.append(combined_im)
        examples["pixel_values"] = combined_pixel_values
        return examples

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        return_d = {"pixel_values": pixel_values}
        return_d[caption_column] = [example[caption_column] for example in examples]

        return return_d
         
    #### END PREPROCESSING/COLLATION ####

    # Set the training transforms
    train_dataset = dataset[args.split].with_transform(preprocess_train)

    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    
    #### START ACCELERATOR PREP ####
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

        
    # Move text_encode and vae to cpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    ref_unet.to(accelerator.device, dtype=weight_dtype)

    # Offload to cpu to redue the GPU memory consumption
    vae = accelerate.cpu_offload(vae)
    text_encoder_one = accelerate.cpu_offload(text_encoder_one)
    text_encoder_two = accelerate.cpu_offload(text_encoder_two)
    ref_unet = accelerate.cpu_offload(ref_unet)

    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.project_name, tracker_config)

    # Training initialization
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0


    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
        

    # Bram Note: This was pretty janky to wrangle to look proper but works to my liking now
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

        
    #### START MAIN TRAINING LOOP #####
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        unwrap_unet = accelerator.unwrap_model(unet)
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step and (not args.hard_skip_resume):
                if step % args.gradient_accumulation_steps == 0:
                    print(f"Dummy processing step {step}, will start training at {resume_step}")
                continue
            with accelerator.accumulate(unet):

                feed_pixel_values = torch.cat(batch["pixel_values"].chunk(2, dim=1))
                
                with torch.no_grad():
                    latents = vae.encode(feed_pixel_values.to(weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)

                bsz = latents.shape[0]

                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                
                timesteps = timesteps.chunk(2)[0].repeat(2)
                noise = noise.chunk(2)[0].repeat(2, 1, 1, 1)


                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                with torch.no_grad():
                    add_time_ids = torch.tensor([args.resolution, 
                                                    args.resolution,
                                                    0,
                                                    0,
                                                    args.resolution, 
                                                    args.resolution],
                                                    dtype=weight_dtype,
                                                    device=accelerator.device)[None, :].repeat(timesteps.size(0), 1)
                    prompt_batch = encode_prompt_sdxl(batch, 
                                                        text_encoders,
                                                        tokenizers,
                                                        0.0, 
                                                        caption_column='prompt',
                                                        is_train=True,
                                                        )
                
                prompt_batch["prompt_embeds"] = prompt_batch["prompt_embeds"].repeat(2, 1, 1)
                prompt_batch["pooled_prompt_embeds"] = prompt_batch["pooled_prompt_embeds"].repeat(2, 1)
                unet_added_conditions = {"time_ids": add_time_ids, "text_embeds": prompt_batch["pooled_prompt_embeds"]}
                #### END PREP BATCH ####
                        
                assert noise_scheduler.config.prediction_type == "epsilon"
                target = noise
               
                # Make the prediction from the model we're learning
                model_batch_args = (noisy_latents, timesteps,  prompt_batch["prompt_embeds"])
                
                model_batch_args_win = list(_arg.chunk(2)[0] for _arg in model_batch_args)
                model_batch_args_lose = list(_arg.chunk(2)[1] for _arg in model_batch_args)

                null_prompt_flag = False
                if random.uniform(0, 1) < args.puncond:
                    null_prompt_flag = True
                
                if null_prompt_flag:
                    model_batch_args_lose[-1] = torch.zeros_like(model_batch_args_lose[-1])

                added_cond_kwargs = unet_added_conditions

                assert added_cond_kwargs is not None
                added_cond_kwargs_win = {k : v.chunk(2)[0] for k,v in added_cond_kwargs.items()}
                added_cond_kwargs_lose = {k : v.chunk(2)[1] for k,v in added_cond_kwargs.items()}
                if null_prompt_flag:
                    added_cond_kwargs_lose['text_embeds'] = torch.zeros_like(added_cond_kwargs_lose['text_embeds'])
                
                model_pred_win = unwrap_unet["win"](
                                *model_batch_args_win,
                                 added_cond_kwargs = added_cond_kwargs_win
                                 ).sample
                
                model_pred_lose = unwrap_unet["lose"](
                                *model_batch_args_lose,
                                 added_cond_kwargs = added_cond_kwargs_lose
                                 ).sample

                model_losses_w = (model_pred_win - target.chunk(2)[0]).pow(2).mean(dim=[1,2,3])
                model_losses_l = (model_pred_lose - target.chunk(2)[1]).pow(2).mean(dim=[1,2,3])
                
                ref_model_batch_args = [torch.cat([win_arg, lose_arg], dim=0) for win_arg, lose_arg in zip(model_batch_args_win, model_batch_args_lose)]
                ref_model_added_cond_kwargs = {k : torch.cat([added_cond_kwargs_win[k], added_cond_kwargs_lose[k]], dim=0) for k in added_cond_kwargs}

                ref_pred = ref_unet(
                                *ref_model_batch_args,
                                added_cond_kwargs = ref_model_added_cond_kwargs
                                ).sample.detach()
                
                ref_losses = (ref_pred - target).pow(2).mean(dim=[1,2,3])
                ref_losses_w, ref_losses_l = ref_losses.chunk(2)

                
                win_diff = model_losses_w - ref_losses_w
                lose_diff = model_losses_l - ref_losses_l

                scale_term = -1.0 * args.beta_dpo
                inside_term = scale_term * (win_diff + lose_diff)
                loss = -1 * F.logsigmoid(inside_term).mean()


                #### for log
                win_diff_loss = win_diff.mean()
                lose_diff_loss = lose_diff.mean()
                total_diff_loss = (win_diff + lose_diff).mean()
                raw_model_loss = ((model_losses_w + model_losses_l) / 2.0).mean()
                raw_ref_loss = ref_losses.mean()


                accumulated_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
                avg_win_diff_loss = accelerator.gather(win_diff_loss.repeat(args.train_batch_size)).mean().item()
                avg_lose_diff_loss = accelerator.gather(lose_diff_loss.repeat(args.train_batch_size)).mean().item()
                avg_total_diff_loss = accelerator.gather(total_diff_loss.repeat(args.train_batch_size)).mean().item()
                avg_raw_model_loss = accelerator.gather(raw_model_loss.repeat(args.train_batch_size)).mean().item()
                avg_raw_ref_loss = accelerator.gather(raw_ref_loss.repeat(args.train_batch_size)).mean().item()


                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if not args.use_adafactor: # Adafactor does itself, maybe could do here to cut down on code
                        accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has just performed an optimization step, if so do "end of batch" logging
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                    if global_step % args.checkpointing_steps == 0:

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        try:
                            accelerator.save_state(save_path)
                        except FileExistsError as e:
                            logger.info(e)
                        logger.info(f"Saved state to {save_path}")

                        if torch.distributed.is_initialized():
                            torch.distributed.barrier()

            logs = {"accumulated_loss": accumulated_loss, 
                    "lr": lr_scheduler.get_last_lr()[0],
                    "avg_win_diff_loss " : avg_win_diff_loss,
                    "avg_lose_diff_loss" : avg_lose_diff_loss,
                    "avg_total_diff_loss" : avg_total_diff_loss,
                    "avg_raw_model_loss" : avg_raw_model_loss,
                    "avg_raw_ref_loss" : avg_raw_ref_loss
                    }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
