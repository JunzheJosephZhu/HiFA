SESSION="$1"
export CUDA_VISIBLE_DEVICES=$SESSION


if [ $SESSION -eq 3 ]; then echo ""
    # python textual-inversion/textual_inversion.py --output_dir="gt_images/fish" --train_data_dir="raw_input/no_bg/fish"  --initializer_token="fish" --placeholder_token="_fish_placeholder_" --pretrained_model_name_or_path="SG161222/Realistic_Vision_V5.1_noVAE" --learnable_property="object" --resolution=256 --train_batch_size=1 --gradient_accumulation_steps=4 --max_train_steps=5000 --learning_rate=5.0e-4 --scale_lr --lr_scheduler="constant" --lr_warmup_steps=0 --use_augmentations
    python main.py --text "<token>"  --learned_embeds_path "gt_images/fish/learned_embeds.bin" --image_path "gt_images/fish/0.png" --workspace "trials_fish_gtrate=.5_v8_512" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.5 --h 512 --w 512 --denoising_steps 1
    # python main.py --text "<token>"  --learned_embeds_path "gt_images/fish/learned_embeds.bin" --image_path "gt_images/fish/0.png" --workspace "trials_fish_gtrate=0.3_v8" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.3 --h 256 --w 256 --denoising_steps 1
    # python main.py --text "<token>"  --learned_embeds_path "gt_images/fish/learned_embeds.bin" --image_path "gt_images/fish/0.png" --workspace "trials_fish_gtrate=0.5_anneal=0.7_v8" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.5 --h 256 --w 256 --anneal_gt 0.7 --denoising_steps 1
    # python main.py --text "<token>"  --learned_embeds_path "gt_images/fish/learned_embeds.bin" --image_path "gt_images/fish/0.png" --workspace "trials_fish_gtrate=0.1_v8" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.1 --h 256 --w 256 --denoising_steps 1

fi
exit


if [ $SESSION -eq 1 ]; then echo ""
    python textual-inversion/textual_inversion.py --output_dir="gt_images/teapot" --train_data_dir="raw_input/no_bg/teapot"  --initializer_token="teapot"  --placeholder_token="_teapot_placeholder_" --pretrained_model_name_or_path="SG161222/Realistic_Vision_V5.1_noVAE" --learnable_property="object"  --resolution=256 --train_batch_size=1 --gradient_accumulation_steps=4 --max_train_steps=5000 --learning_rate=5.0e-4 --scale_lr --lr_scheduler="constant" --lr_warmup_steps=0 --use_augmentations
    python main.py --text "<token>" --learned_embeds_path "gt_images/teapot/learned_embeds.bin" --image_path "gt_images/teapot/0.png" --workspace "trials_teapot_gtrate=0.5_v8" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.5 --h 256 --w 256 --denoising_steps 1
    python main.py --text "<token>" --learned_embeds_path "gt_images/teapot/learned_embeds.bin" --image_path "gt_images/teapot/0.png" --workspace "trials_teapot_gtrate=0.3_v8" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.3 --h 256 --w 256 --denoising_steps 1
    python main.py --text "<token>" --learned_embeds_path "gt_images/teapot/learned_embeds.bin" --image_path "gt_images/teapot/0.png" --workspace "trials_teapot_gtrate=0.5_anneal=0.7_v8" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.5 --h 256 --w 256 --anneal_gt 0.7  --denoising_steps 1
    python main.py --text "<token>" --learned_embeds_path "gt_images/teapot/learned_embeds.bin" --image_path "gt_images/teapot/0.png" --workspace "trials_teapot_gtrate=0.1_v8" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.1 --h 256 --w 256 --denoising_steps 1
fi

if [ $SESSION -eq 2 ]; then echo ""
    python textual-inversion/textual_inversion.py --output_dir="gt_images/dragon" --train_data_dir="raw_input/no_bg/dragon"  --initializer_token="dragon"  --placeholder_token="_dragon_placeholder_"  --pretrained_model_name_or_path="SG161222/Realistic_Vision_V5.1_noVAE" --learnable_property="object" --resolution=256 --train_batch_size=1 --gradient_accumulation_steps=4 --max_train_steps=5000 --learning_rate=5.0e-4 --scale_lr --lr_scheduler="constant" --lr_warmup_steps=0 --use_augmentations
    python main.py --text "<token>" --learned_embeds_path "gt_images/dragon/learned_embeds.bin" --image_path "gt_images/dragon/0.png" --workspace "trials_dragon_gtrate=0.5_v8" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.5 --h 256 --w 256 --denoising_steps 1
    python main.py --text "<token>" --learned_embeds_path "gt_images/dragon/learned_embeds.bin" --image_path "gt_images/dragon/0.png" --workspace "trials_dragon_gtrate=0.5_anneal=0.7_v8" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.5 --h 256 --w 256 --anneal_gt 0.7 --denoising_steps 1
    python main.py --text "<token>" --learned_embeds_path "gt_images/dragon/learned_embeds.bin" --image_path "gt_images/dragon/0.png" --workspace "trials_dragon_gtrate=0.1_v8" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.5 --h 256 --w 256  --denoising_steps 1
   fi

if [ $SESSION -eq 3 ]; then echo ""
    # python textual-inversion/textual_inversion.py --output_dir="gt_images/dinosaur" --train_data_dir="raw_input/no_bg/dinosaur" --initializer_token="dinosaur"  --placeholder_token="_dinosaur_placeholder_" --pretrained_model_name_or_path="SG161222/Realistic_Vision_V5.1_noVAE" --learnable_property="object"  --resolution=256 --train_batch_size=1 --gradient_accumulation_steps=4 --max_train_steps=5000 --learning_rate=5.0e-4 --scale_lr --lr_scheduler="constant" --lr_warmup_steps=0 --use_augmentations
    python main.py --text "<token>" --learned_embeds_path "gt_images/dinosaur/learned_embeds.bin" --image_path "gt_images/dinosaur/0.png" --workspace "trials_dinosaur_gtrate=0.5_v8" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.1 --h 256 --w 256 --denoising_steps 1
    python main.py --text "<token>" --learned_embeds_path "gt_images/dinosaur/learned_embeds.bin" --image_path "gt_images/dinosaur/0.png" --workspace "trials_dinosaur_gtrate=0.5_anneal=0.7_v8" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.5 --h 256 --w 256 --anneal_gt 0.7  --denoising_steps 1
    python main.py --text "<token>" --learned_embeds_path "gt_images/dinosaur/learned_embeds.bin" --image_path "gt_images/dinosaur/0.png" --workspace "trials_dinosaur_gtrate=0.1_v8" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.1 --h 256 --w 256  --denoising_steps 1

fi



if [ $SESSION -eq 4 ]; then echo ""
    SESSION=2
    export CUDA_VISIBLE_DEVICES=$SESSION

    # python textual-inversion/textual_inversion.py --output_dir="gt_images/fish" --train_data_dir="raw_input/no_bg/fish"  --initializer_token="fish" --placeholder_token="_fish_placeholder_" --pretrained_model_name_or_path="SG161222/Realistic_Vision_V5.1_noVAE" --learnable_property="object" --resolution=256 --train_batch_size=1 --gradient_accumulation_steps=4 --max_train_steps=5000 --learning_rate=5.0e-4 --scale_lr --lr_scheduler="constant" --lr_warmup_steps=0 --use_augmentations
    python main.py --text "<token>"  --learned_embeds_path "gt_images/fish/learned_embeds.bin" --image_path "gt_images/fish/0.png" --workspace "trials_fish_gtrate=1_anneal=0.7_gt_v8" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 1 --h 256 --w 256 --anneal_gt 0.7 --denoising_steps 1 
    python main.py --text "<token>" --learned_embeds_path "gt_images/dragon/learned_embeds.bin" --image_path "gt_images/dragon/0.png" --workspace "trials_dragon_gtrate=1_anneal=0.7_v8" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 1 --h 256 --w 256 --anneal_gt 0.7 --denoising_steps 1
    python main.py --text "<token>" --learned_embeds_path "gt_images/teapot/learned_embeds.bin" --image_path "gt_images/teapot/0.png" --workspace "trials_teapot_gtrate=1_anneal=0.7_v8" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 1 --h 256 --w 256 --anneal_gt 0.7  --denoising_steps 1

fi

# if [ $SESSION -eq 0 ]; then echo ""
#     python main.py --text "a painted teapot" --workspace "trials_teapot_gtrate=0.5_gt0.1_latent1" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.5 --h 256 --w 256 --image_path "gt_images/teapot/0.png" --denoising_steps 1 --lambda_gt 0.1 --lambda_latent 1
#     python main.py --text "a painted teapot" --workspace "trials_teapot_gtrate=0.5_gt1_latent.1" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.5 --h 256 --w 256 --image_path "gt_images/teapot/0.png" --denoising_steps 1 --lambda_gt 1 --lambda_latent 0.1
#     python main.py --text "a painted teapot" --workspace "trials_teapot_gtrate=0.8_gt0.1_latent1" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.8 --h 256 --w 256 --image_path "gt_images/teapot/0.png" --denoising_steps 1 --lambda_gt 0.1 --lambda_latent 1
#     python main.py --text "a painted teapot" --workspace "trials_teapot_gtrate=0.8_gt1_latent.1" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.8 --h 256 --w 256 --image_path "gt_images/teapot/0.png" --denoising_steps 1 --lambda_gt 1 --lambda_latent 0.1
#     python main.py --text "a painted teapot" --workspace "trials_teapot_gtrate=0.5_gt0.5_latent0.5" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.5 --h 256 --w 256 --image_path "gt_images/teapot/0.png" --denoising_steps 1 --lambda_gt 0.5 --lambda_latent 0.5
#     python main.py --text "a painted teapot" --workspace "trials_teapot_gtrate=0.8_gt0.5_latent0.5" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.8 --h 256 --w 256 --image_path "gt_images/teapot/0.png" --denoising_steps 1 --lambda_gt 0.5 --lambda_latent 0.5
# fi


# if [ $SESSION -eq 1 ]; then echo ""
#     python main.py --text "a stone dragon statue" --workspace "trials_dragon_gtrate=0.5_gt0.1_latent1" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.5 --h 256 --w 256 --image_path "gt_images/dragon/0.png" --denoising_steps 1 --lambda_gt 0.1 --lambda_latent 1
#     python main.py --text "a stone dragon statue" --workspace "trials_dragon_gtrate=0.5_gt1_latent.1" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.5 --h 256 --w 256 --image_path "gt_images/dragon/0.png" --denoising_steps 1 --lambda_gt 1 --lambda_latent 0.1
#     python main.py --text "a stone dragon statue" --workspace "trials_dragon_gtrate=0.8_gt0.1_latent1" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.8 --h 256 --w 256 --image_path "gt_images/dragon/0.png" --denoising_steps 1 --lambda_gt 0.1 --lambda_latent 1
#     python main.py --text "a stone dragon statue" --workspace "trials_dragon_gtrate=0.8_gt1_latent.1" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.8 --h 256 --w 256 --image_path "gt_images/dragon/0.png" --denoising_steps 1 --lambda_gt 1 --lambda_latent 0.1
#     python main.py --text "a stone dragon statue" --workspace "trials_dragon_gtrate=0.5_gt0.5_latent0.5" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.5 --h 256 --w 256 --image_path "gt_images/dragon/0.png" --denoising_steps 1 --lambda_gt 0.5 --lambda_latent 0.5
#     python main.py --text "a stone dragon statue" --workspace "trials_dragon_gtrate=0.8_gt0.5_latent0.5" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.8 --h 256 --w 256 --image_path "gt_images/dragon/0.png" --denoising_steps 1 --lambda_gt 0.5 --lambda_latent 0.5
# fi

# if [ $SESSION -eq 2 ]; then echo ""
#     python main.py --text "a ficus tree planted in a pot" --workspace "trials_ficus_gtrate=0.5_gt0.1_latent1" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.5 --h 256 --w 256 --image_path "gt_images/ficus_nogt/0.png" --denoising_steps 1 --lambda_gt 0.1 --lambda_latent 1
#     python main.py --text "a ficus tree planted in a pot" --workspace "trials_ficus_gtrate=0.5_gt1_latent.1" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.5 --h 256 --w 256 --image_path "gt_images/ficus_nogt/0.png" --denoising_steps 1 --lambda_gt 1 --lambda_latent 0.1
#     python main.py --text "a ficus tree planted in a pot" --workspace "trials_ficus_gtrate=0.8_gt0.1_latent1" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.8 --h 256 --w 256 --image_path "gt_images/ficus_nogt/0.png" --denoising_steps 1 --lambda_gt 0.1 --lambda_latent 1
#     python main.py --text "a ficus tree planted in a pot" --workspace "trials_ficus_gtrate=0.8_gt1_latent.1" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.8 --h 256 --w 256 --image_path "gt_images/ficus_nogt/0.png" --denoising_steps 1 --lambda_gt 1 --lambda_latent 0.1
#     python main.py --text "a ficus tree planted in a pot" --workspace "trials_ficus_gtrate=0.5_gt0.5_latent0.5" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.5 --h 256 --w 256 --image_path "gt_images/ficus_nogt/0.png" --denoising_steps 1 --lambda_gt 0.5 --lambda_latent 0.5
#     python main.py --text "a ficus tree planted in a pot" --workspace "trials_ficus_gtrate=0.8_gt0.5_latent0.5" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.8 --h 256 --w 256 --image_path "gt_images/ficus_nogt/0.png" --denoising_steps 1 --lambda_gt 0.5 --lambda_latent 0.5
# fi

# if [ $SESSION -eq 3 ]; then echo ""
#     python main.py --text "a red-color drum set" --workspace "trials_drum_gtrate=0.5_gt0.1_latent1" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.5 --h 256 --w 256 --image_path "gt_images/drum_nogt/0.png" --denoising_steps 1 --lambda_gt 0.1 --lambda_latent 1
#     python main.py --text "a red-color drum set" --workspace "trials_drum_gtrate=0.5_gt1_latent.1" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.5 --h 256 --w 256 --image_path "gt_images/drum_nogt/0.png" --denoising_steps 1 --lambda_gt 1 --lambda_latent 0.1
#     python main.py --text "a red-color drum set" --workspace "trials_drum_gtrate=0.8_gt0.1_latent1" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.8 --h 256 --w 256 --image_path "gt_images/drum_nogt/0.png" --denoising_steps 1 --lambda_gt 0.1 --lambda_latent 1
#     python main.py --text "a red-color drum set" --workspace "trials_drum_gtrate=0.8_gt1_latent.1" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.8 --h 256 --w 256 --image_path "gt_images/drum_nogt/0.png" --denoising_steps 1 --lambda_gt 1 --lambda_latent 0.1
#     python main.py --text "a red-color drum set" --workspace "trials_drum_gtrate=0.5_gt0.5_latent0.5" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.5 --h 256 --w 256 --image_path "gt_images/drum_nogt/0.png" --denoising_steps 1 --lambda_gt 0.5 --lambda_latent 0.5
#     python main.py --text "a red-color drum set" --workspace "trials_drum_gtrate=0.8_gt0.5_latent0.5" --dir_text --albedo --min_percent 0.3 --sd_version 5.1 --seed 810 --dir_rate 0.5 --gt_image_rate 0.8 --h 256 --w 256 --image_path "gt_images/drum_nogt/0.png" --denoising_steps 1 --lambda_gt 0.5 --lambda_latent 0.5
# fi


exit
