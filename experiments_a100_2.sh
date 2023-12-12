SESSION="$1"
export CUDA_VISIBLE_DEVICES=$SESSION


if [ $SESSION -eq 0 ]; then echo ""
    python main.py --text "a mic" --workspace "trials_mic_gtrate=0.5_gt0.1_latent1" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.5 --h 256 --w 256 --image_path "gt_images/mic_nogt/0.png" --denoising_steps 1 --lambda_gt 0.1 --lambda_latent 1
    python main.py --text "a mic" --workspace "trials_mic_gtrate=0.5_gt1_latent.1" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.5 --h 256 --w 256 --image_path "gt_images/mic_nogt/0.png" --denoising_steps 1 --lambda_gt 1 --lambda_latent 0.1
    python main.py --text "a mic" --workspace "trials_mic_gtrate=0.8_gt0.1_latent1" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.8 --h 256 --w 256 --image_path "gt_images/mic_nogt/0.png" --denoising_steps 1 --lambda_gt 0.1 --lambda_latent 1
    python main.py --text "a mic" --workspace "trials_mic_gtrate=0.8_gt1_latent.1" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.8 --h 256 --w 256 --image_path "gt_images/mic_nogt/0.png" --denoising_steps 1 --lambda_gt 1 --lambda_latent 0.1
    python main.py --text "a mic" --workspace "trials_mic_gtrate=0.5_gt0.5_latent0.5" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.5 --h 256 --w 256 --image_path "gt_images/mic_nogt/0.png" --denoising_steps 1 --lambda_gt 0.5 --lambda_latent 0.5
    python main.py --text "a mic" --workspace "trials_mic_gtrate=0.8_gt0.5_latent0.5" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.8 --h 256 --w 256 --image_path "gt_images/mic_nogt/0.png" --denoising_steps 1 --lambda_gt 0.5 --lambda_latent 0.5
fi


if [ $SESSION -eq 1 ]; then echo ""
    python main.py --text "a stone dragon statue" --workspace "trials_dragon_gtrate=0.5_gt0_latent1" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.5 --h 256 --w 256 --image_path "gt_images/dragon/0.png" --denoising_steps 1 --lambda_gt 0. --lambda_latent 1
    python main.py --text "a red-color drum set" --workspace "trials_drum_gtrate=0.5_gt0_latent1" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.5 --h 256 --w 256 --image_path "gt_images/drum_nogt/0.png" --denoising_steps 1 --lambda_gt 0. --lambda_latent 1
    python main.py --text "a painted teapot" --workspace "trials_teapot_gtrate=0.5_gt0._latent1" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.5 --h 256 --w 256 --image_path "gt_images/teapot/0.png" --denoising_steps 1 --lambda_gt 0. --lambda_latent 1
    python main.py --text "a mic" --workspace "trials_mic_gtrate=0.5_gt1_latent.1" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.5 --h 256 --w 256 --image_path "gt_images/mic_nogt/0.png" --denoising_steps 1 --lambda_gt 0. --lambda_latent 0.1
fi

# SESSION="$1"
# export CUDA_VISIBLE_DEVICES=$SESSION


# if [ $SESSION -eq 0 ]; then echo ""
#     python main.py --text "rolling stone tongue and lips logo" --workspace "trials_lips_gtrate=0.5_v4" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.5 --h 256 --w 256 --image_path "gt_images/lips/0.png"
#     python main.py --text "rolling stone tongue and lips logo" --workspace "trials_lips_gtrate=0.5_anneal_v4" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.5 --h 256 --w 256 --anneal_gt --image_path "gt_images/lips/0.png"
#     python main.py --text "rolling stone tongue and lips logo" --workspace "trials_lips_gtrate=0.1_v4" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.1 --gt_image_rate 0.5 --h 256 --w 256 --image_path "gt_images/lips/0.png"
#     python main.py --text "Hircine's Ring" --workspace "trials_ring_gtrate=0.5_v4" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.5 --h 256 --w 256 --image_path "gt_images/ring/0.png"
#     python main.py --text "Hircine's Ring" --workspace "trials_ring_gtrate=0.5_anneal_v4" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.5 --h 256 --w 256 --anneal_gt --image_path "gt_images/ring/0.png"
#     python main.py --text "Hircine's Ring" --workspace "trials_ring_gtrate=0.1_v4" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.1 --gt_image_rate 0.5 --h 256 --w 256 --image_path "gt_images/ring/0.png"

# fi

# if [ $SESSION -eq 5 ]; then echo ""
#     python main.py --text "A tulip" --workspace trials_sd5.1_tulip_seed810_min0.3_v5 --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5
#     python main.py --text "A stack of pancakes covered in maple syrup" --workspace trials_sd5.1_pancake_seed810_min0.3_v4 --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --h 256 --w 256 
#     python main.py --text "A stack of pancakes covered in maple syrup" --workspace trials_sd5.1_pancake_seed810_min0.02_sdslike_v4 --dir_text --albedo --min_percent 0.02 --sd_version 5.1 --seed 810 --dir_rate 0.5 --h 256 --w 256 --iters 15000 --sds_weight "sdslike"
#     python main.py --text "A stack of pancakes covered in maple syrup" --workspace trials_sd5.1_pancake_seed810_min0.02_noisematch_v4 --dir_text --albedo --min_percent 0.02 --sd_version 5.1 --seed 810 --dir_rate 0.5 --h 256 --w 256 --iters 15000 --sds_weight "noisematch"
#     python main.py --text "A stack of pancakes covered in maple syrup" --workspace trials_sd5.1_pancake_seed810_min0.02_strongerlower_v4 --dir_text --albedo --min_percent 0.02 --sd_version 5.1 --seed 810 --dir_rate 0.5 --h 256 --w 256 --iters 15000 --sds_weight "strongerlower"
#     python main.py --text "nemo fish" --workspace "trials_fish_gtrate=0.5_v4" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.5 --h 256 --w 256 --image_path "gt_images/fish/0.png"
#     python main.py --text "nemo fish" --workspace "trials_fish_gtrate=0.5_anneal_v4" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.5 --h 256 --w 256 --anneal_gt --image_path "gt_images/fish/0.png"
#     python main.py --text "nemo fish" --workspace "trials_fish_gtrate=0.1_v4" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.1 --gt_image_rate 0.5 --h 256 --w 256 --image_path "gt_images/fish/0.png"

# fi
# if [ $SESSION -eq 1 ]; then echo ""
#     python main.py --text "toy figure of firekeeper from dark souls 3" --workspace trials_firekeeper_onlygt_v4 --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 1.0 --h 256 --w 256
#     python main.py --text "toy figure of firekeeper from dark souls 3" --workspace "trials_firekeeper_gtrate=0.3_v4" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.3 --h 256 --w 256
#     python main.py --text "toy figure of firekeeper from dark souls 3" --workspace "trials_firekeeper_gtrate=0.3_anneal_v4" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.3 --h 256 --w 256 --anneal_gt
#     python main.py --text "toy figure of firekeeper from dark souls 3" --workspace "trials_firekeeper_gtrate=0.3_lambda_gt=mask=3.0_v4" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.3 --h 256 --w 256 --lambda_gt 3 --lambda_mask 3
#     python main.py --text "toy figure of firekeeper from dark souls 3" --workspace "trials_firekeeper_gtrate=0.3_lambda_gt=mask=0.3_v4" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.3 --h 256 --w 256 --lambda_gt 0.3 --lambda_mask 0.3
#     python main.py --text "toy figure of firekeeper from dark souls 3" --workspace "trials_firekeeper_gtrate=0.5_lambda_gt=mask=3.0_v4" --dir_text --albedo --min_percent 0.5 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.5 --h 256 --w 256  --lambda_gt 3 --lambda_mask 3
#     python main.py --text "toy figure of firekeeper from dark souls 3" --workspace "trials_firekeeper_gtrate=0.5_lambda_gt=mask=3.0_anneal_v4" --dir_text --albedo --min_percent 0.5 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.5 --h 256 --w 256   --lambda_gt 3 --lambda_mask 3 --anneal_gt
# fi

# if [ $SESSION -eq 2 ]; then echo ""
#     python main.py --text "a painted teapot" --workspace "trials_teapot_gtrate=0.3_v4" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.3 --h 256 --w 256 --image_path "gt_images/teapot/0.png" --denoising_steps 1
#     python main.py --text "a painted teapot" --workspace "trials_teapot_gtrate=0.8_v4" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.8 --h 256 --w 256  --image_path "gt_images/teapot/0.png" --denoising_steps 1
#     python main.py --text "a painted teapot" --workspace "trials_teapot_gtrate=0.1_v4" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.1 --h 256 --w 256 --image_path "gt_images/teapot/0.png" --denoising_steps 1
#     # python main.py --text "ironman" --workspace "trials_ironman_gtrate=0.5_v4" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.5 --h 256 --w 256 --image_path "gt_images/ironman/0.png"
#     # python main.py --text "ironman" --workspace "trials_ironman_gtrate=0.5_anneal_v4" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.5 --h 256 --w 256 --anneal_gt --image_path "gt_images/ironman/0.png"
#     # python main.py --text "ironman" --workspace "trials_ironman_gtrate=0.1_v4" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.1 --gt_image_rate 0.5 --h 256 --w 256 --image_path "gt_images/ironman/0.png"
# fi

# if [ $SESSION -eq 3 ]; then echo ""
#     python main.py --text "a stone dragon statue" --workspace "trials_dragon_gtrate=0.5_v4" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.5 --h 256 --w 256 --image_path "gt_images/dragon/0.png"
#     python main.py --text "a stone dragon statue" --workspace "trials_dragon_gtrate=0.5_anneal_v4" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.5 --h 256 --w 256 --anneal_gt --image_path "gt_images/dragon/0.png"
#     python main.py --text "a stone dragon statue" --workspace "trials_dragon_gtrate=0.1_v4" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.1 --gt_image_rate 0.5 --h 256 --w 256 --image_path "gt_images/dragon/0.png"
#     python main.py --text "a toy grabber with dinosaur head" --workspace "trials_dinosaur_gtrate=0.5_v4" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.5 --h 256 --w 256 --image_path "gt_images/dinosaur3/0.png"
#     python main.py --text "a toy grabber with dinosaur head" --workspace "trials_dinosaur_gtrate=0.5_anneal_v4" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.5 --h 256 --w 256 --anneal_gt --image_path "gt_images/dinosaur3/0.png"
#     python main.py --text "a toy grabber with dinosaur head" --workspace "trials_dinosaur_gtrate=0.1_v4" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.1 --gt_image_rate 0.5 --h 256 --w 256 --image_path "gt_images/dinosaur3/0.png"

# fi
# exit
