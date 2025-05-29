

accelerate launch --config_file=config/ac_ds_8gpu_zero0.yaml train.py \
        --pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 \
        --pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix \
        --resolution=1024 \
        --dataloader_num_workers 16 \
        --train_batch_size=1 \
        --gradient_accumulation_steps=16 \
        --max_train_steps=6000 \
        --learning_rate=3e-09 --scale_lr --lr_scheduler=constant_with_warmup --lr_warmup_steps=100 \
        --mixed_precision=bf16 \
        --allow_tf32 \
        --checkpointing_steps=200 \
        --output_dir=output \
        --resume_from_checkpoint latest \
        --use_adafactor \
        --gradient_checkpointing \
        --dataset_name=data-is-better-together/open-image-preferences-v1-binarized \