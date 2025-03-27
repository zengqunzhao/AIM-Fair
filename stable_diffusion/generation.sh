CUDA_VISIBLE_DEVICES='0' python scripts/txt2img.py \
    --outdir "./generated_images_CelebA" \
    --ckpt "./v2-1_512-ema-pruned.ckpt" \
    --n_samples 15 \
    --n_iter 14 \
    --prompt-type "Multiply_Prompt" \
    --target-sensitive "Smiling_Male" \

    # Smiling_Male
    # Smiling_Young



