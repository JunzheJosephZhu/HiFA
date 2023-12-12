#!/bin/bash
# python sd_generate.py
# python main.py --text "matte painting of a castle made of cheesecake surrounded by a moat made of ice cream" --workspace castle31 --fp16 --dir_text --render_latents --render_fg_variance --loss_type l2 --albedo --lambda_entropy 1000 --iters 30000 --lambda_gan 1000.0 --gan_iters 5000 --max_scale 0.25 --min_scale 0.125 --scale_anneal 0.02 --lr_discriminator 1e-3 --reg_param 1.0
# python main.py --text "matte painting of a castle made of cheesecake surrounded by a moat made of ice cream" --workspace castle32 --fp16 --dir_text --render_latents --render_fg_variance --loss_type l2 --albedo --lambda_entropy 1000 --iters 30000 --lambda_gan 10000.0 --gan_iters 5000 --max_scale 0.25 --min_scale 0.125 --scale_anneal 0.02 --lr_discriminator 1e-3 --reg_param 1.0
# python main.py --text "matte painting of a castle made of cheesecake surrounded by a moat made of ice cream" --workspace castle33 --fp16 --dir_text --render_latents --render_fg_variance --loss_type l2 --albedo --lambda_entropy 1000 --iters 30000 --lambda_gan 1000.0 --gan_iters 5000 --max_scale 0.25 --min_scale 0.125 --scale_anneal 0.02 --lr_discriminator 1e-3 --reg_param 1.0 --gan_imsize 32 --gan_on_latents

# python main.py --text "matte painting of a castle made of cheesecake surrounded by a moat made of ice cream" --workspace castle104_3 --fp16 --dir_text --albedo --lambda_entropy 1e-4 --iters 29000 --lambda_depth 1 --guidance_scale 100 --progressive --h 256 --w 256
# python main.py --text "matte painting of a castle made of cheesecake surrounded by a moat made of ice cream" --workspace castle104_rgb_4 --fp16 --dir_text --albedo --lambda_entropy 1e-4 --iters 29000 --lambda_depth 1 --guidance_scale 100 --progressive --h 256 --w 256 --lambda_rgb 1.0


# python main.py --text "matte painting of a castle made of cheesecake surrounded by a moat made of ice cream" --workspace castle104_rgb_5 --fp16 --dir_text --albedo --lambda_entropy 1e-4 --iters 40000 --lambda_depth 1 --guidance_scale 100 --rgb_guidance_scale 15 --progressive --h 256 --w 256 --lambda_rgb 1.0 --min_percent 0.2
# python main.py --text "matte painting of a castle made of cheesecake surrounded by a moat made of ice cream" --workspace castle104_depthcond --fp16 --dir_text --albedo --lambda_entropy 1e-4 --iters 40000 --lambda_depth 1 --guidance_scale 100 --rgb_guidance_scale 100 --progressive --h 256 --w 256 --min_percent 0.2 --use_depth_conditioning

# python main.py --text "matte painting of a castle made of cheesecake surrounded by a moat made of ice cream" --workspace castle104_rgb=0.1_depthreg=0.1 --fp16 --dir_text --albedo --lambda_entropy 1e-4 --iters 40000 --lambda_depth 1 --guidance_scale 100 --rgb_guidance_scale 100 --progressive --h 256 --w 256 --lambda_rgb 0.1 --min_percent 0.2 --lambda_depthreg 0.1
# python main.py --text "matte painting of a castle made of cheesecake surrounded by a moat made of ice cream" --workspace castle104_rgb_4 --fp16 --dir_text --albedo --lambda_entropy 1e-4 --iters 40000 --lambda_depth 1 --guidance_scale 100 --rgb_guidance_scale 100 --progressive --h 256 --w 256 --lambda_rgb 1.0 --min_percent 0.2
# python main.py --text "matte painting of a castle made of cheesecake surrounded by a moat made of ice cream" --workspace castle104_rgb=1_depthreg=0.1 --fp16 --dir_text --albedo --lambda_entropy 1e-4 --iters 40000 --lambda_depth 1 --guidance_scale 100 --rgb_guidance_scale 100 --progressive --h 256 --w 256 --lambda_rgb 1.0 --min_percent 0.2 --lambda_depthreg 0.1
# python main.py --text "matte painting of a castle made of cheesecake surrounded by a moat made of ice cream" --workspace castle104_depthreg=0.1 --fp16 --dir_text --albedo --lambda_entropy 1e-4 --iters 40000 --lambda_depth 1 --guidance_scale 100 --rgb_guidance_scale 100 --progressive --h 256 --w 256 --lambda_rgb 0.0 --min_percent 0.2 --lambda_depthreg 0.1

# failed to converge
# python main.py --text "The leaning tower of Pisa" --workspace tower104_rgb=0.1_depthreg=0.1_annealdepth --fp16 --dir_text --albedo --lambda_entropy 1e-4 --iters 40000 --lambda_depth 1 --guidance_scale 100 --rgb_guidance_scale 100 --progressive --h 256 --w 256 --lambda_rgb 0.1 --min_percent 0.2 --lambda_depthreg 0.1 --anneal_depthreg_weight

# try using more ddim steps
# python main.py --text "detailed stone head of superman" --workspace head104_rgb=0.1_rgb_guidance=7.5_depthreg=0.0_annealdepth_step=2 --fp16 --dir_text --albedo --lambda_entropy 1e-4 --iters 40000 --lambda_depth 1 --guidance_scale 7.5 --rgb_guidance_scale 7.5 --progressive --h 256 --w 256 --lambda_rgb 0.1 --min_percent 0.2 --lambda_depthreg 0.0 --anneal_depthreg_weight --suppress_face --denoising_steps 2


# python main.py --text "detailed stone head of superman" --workspace head104_rgb=0.1_depthreg=0.1_annealdepth_step=3 --fp16 --dir_text --albedo --iters 40000 --lambda_depth 1  --progressive --h 256 --w 256 --lambda_rgb 0.1 --min_percent 0.2 --lambda_depthreg 0.1 --anneal_depthreg_weight --default_polar 60 --denoising_steps 3

# refactor code to add text z interpolation
# python main.py --text "detailed stone head of superman" --workspace head104_rgb=0.1_depthreg=0.1_annealdepth --fp16 --dir_text --albedo --iters 20000 --lambda_depth 1  --progressive --h 64 --w 64 --lambda_rgb 0.1 --min_percent 0.43 --lambda_depthreg 0.1 --anneal_depthreg_weight --default_polar 60
# python main.py --text "detailed stone head of superman" --workspace head104_rgb=0.1_depthreg=0.1_annealdepth_finetune --fp16 --dir_text --albedo --iters 40000 --lambda_depth 1  --progressive --h 256 --w 256 --lambda_rgb 0.1 --min_percent 0.2 --lambda_depthreg 0.1 --anneal_depthreg_weight --default_polar 60 --init_with head104_rgb=0.1_depthreg=0.1_annealdepth --test

# compare the following two as ablation study for steps
# python main.py --text "detailed stone head of superman" --workspace head104_rgb=0.1_depthreg=0.1_annealdepth_256x256_step=2 --fp16 --dir_text --albedo --iters 40000 --lambda_depth 1 --progressive --h 256 --w 256 --lambda_rgb 0.1 --min_percent 0.2 --lambda_depthreg 0.1 --anneal_depthreg_weight --default_polar 60 --denoising_steps 2

# python main.py --text "detailed stone head of superman" --workspace head104_rgb=0.1_depthreg=0.1_annealdepth_256x256 --fp16 --dir_text --albedo --iters 40000 --lambda_depth 1 --progressive --h 256 --w 256 --lambda_rgb 0.1 --min_percent 0.2 --lambda_depthreg 0.1 --anneal_depthreg_weight --default_polar 60

# ablation study for removing rgb
# python main.py --text "detailed stone head of superman" --workspace head104_rgb=0.0_depthreg=0.1_annealdepth_256x256 --fp16 --dir_text --albedo --iters 40000 --lambda_depth 1 --progressive --h 256 --w 256 --lambda_rgb 0.0 --min_percent 0.2 --lambda_depthreg 0.1 --anneal_depthreg_weight --default_polar 60

# maybe add negative prompt for scarf and crest

# python main.py --text "detailed stone bust of batman" --workspace bat104_rgb=0.1_depthreg=0.1_annealdepth_256x256 --fp16 --dir_text --albedo --iters 40000 --lambda_depth 1 --progressive --h 256 --w 256 --lambda_rgb 0.1 --min_percent 0.2 --lambda_depthreg 0.1 --anneal_depthreg_weight --default_polar 60

# python main.py --text "A blue poison-dart frog sitting on a water lily" --workspace "/home/joseph/Dropbox/4090_backup/stable-dreamfusion-main/trial_450/trial_final_frog_step=4_eta=1.0" --fp16 --dir_text --albedo --iters 10000 --lambda_depth 3 --progressive --h 512 --w 512 --H 512 --W 512 --lambda_rgb 0.1 --min_percent 0.2 --lambda_depthreg 0.1 --anneal_depthreg_weight --default_polar 70 --min_lr 1e-3 --dir_rate 0.5 --denoising_steps 4 --lambda_orient 0.0 --lambda_monotonic 0.1 --eta 1.0 --use_scaler --test --save_mesh

# python main.py --text "a pyre made from bone, no fire" --workspace "/home/joseph/Dropbox/4090_backup/stable-dreamfusion-main/trial_ulti_pyre" --fp16 --dir_text --albedo --iters 10000 --lambda_depth 3 --progressive --h 512 --w 512 --H 512 --W 512 --lambda_rgb 0.1 --min_percent 0.2 --lambda_depthreg 0.1 --anneal_depthreg_weight --default_polar 70 --min_lr 1e-3 --dir_rate 0.5 --denoising_steps 4 --lambda_orient 0.0 --lambda_monotonic 0.1 --eta 1.0 --use_scaler --test --save_mesh

# python main.py --text "A wooden buddha head with many faces next to each other" --workspace "/home/joseph/4090_backup/stable-dreamfusion-main/trial_ulti_buddha(next)" --fp16 --dir_text --albedo --iters 10000 --lambda_depth 3 --progressive --h 512 --w 512 --H 512 --W 512 --lambda_rgb 0.1 --min_percent 0.2 --lambda_depthreg 0.1 --anneal_depthreg_weight --default_polar 70 --min_lr 1e-3 --dir_rate 0.5 --denoising_steps 4 --lambda_orient 0.0 --lambda_monotonic 0.1 --eta 1.0 --use_scaler --test --write_image #--save_mesh
# python main.py --text "Michelangelo style colorful statue of an astronaut" --workspace "/home/joseph/4090_backup/stable-dreamfusion-main/trial_ulti_astronaut(colorful)" --fp16 --dir_text --albedo --iters 10000 --lambda_depth 3 --progressive --h 512 --w 512 --H 512 --W 512 --lambda_rgb 0.1 --min_percent 0.2 --lambda_depthreg 0.1 --anneal_depthreg_weight --default_polar 70 --min_lr 1e-3 --dir_rate 0.5 --denoising_steps 4 --lambda_orient 0.0 --lambda_monotonic 0.1 --eta 1.0 --use_scaler --test #--save_mesh
# python main.py --text "a beautiful peacock" --workspace "/home/joseph/4090_backup/stable-dreamfusion-main/trial_ulti_peacock(nophoto)" --fp16 --dir_text --albedo --iters 10000 --lambda_depth 3 --progressive --h 512 --w 512 --H 512 --W 512 --lambda_rgb 0.1 --min_percent 0.2 --lambda_depthreg 0.1 --anneal_depthreg_weight --default_polar 70 --min_lr 1e-3 --dir_rate 0.5 --denoising_steps 4 --lambda_orient 0.0 --lambda_monotonic 0.1 --eta 1.0 --use_scaler --test #--save_mesh
# python main.py --text "a stack of pancakes covered in maple syrup" --workspace "/home/joseph/4090_backup/stable-dreamfusion-main/trial_ulti_pancake" --fp16 --dir_text --albedo --iters 10000 --lambda_depth 3 --progressive --h 512 --w 512 --H 512 --W 512 --lambda_rgb 0.1 --min_percent 0.2 --lambda_depthreg 0.1 --anneal_depthreg_weight --default_polar 70 --min_lr 1e-3 --dir_rate 0.5 --denoising_steps 4 --lambda_orient 0.0 --lambda_monotonic 0.1 --eta 1.0 --use_scaler --test #--save_mesh
# python main.py --text "a highly detailed stone bust of Theodoros Kolokotronis" --workspace "/home/joseph/4090_backup/stable-dreamfusion-main/trial_ulti_theo_seed=123456" --fp16 --dir_text --albedo --iters 10000 --lambda_depth 3 --progressive --h 512 --w 512 --H 512 --W 512 --lambda_rgb 0.1 --min_percent 0.2 --lambda_depthreg 0.1 --anneal_depthreg_weight --default_polar 70 --min_lr 1e-3 --dir_rate 0.5 --denoising_steps 4 --lambda_orient 0.0 --lambda_monotonic 0.1 --eta 1.0 --use_scaler --test #--save_mesh
# python main.py --text "A stack of pancakes covered in maple syrup" --workspace "trial_micro_pancake" --fp16 --dir_text --albedo --iters 10000 --lambda_depth 3 --h 256 --w 256 --H 800 --W 800 --lambda_rgb 0.1 --min_percent 0.2 --lambda_depthreg 0.1 --anneal_depthreg_weight --default_polar 60 --min_lr 1e-3 --denoising_steps 4 --lambda_orient 1e-2 --lambda_monotonic 0.1 --eta 1.0 --use_scaler --microfacet
# python main.py --text "A stack of pancakes covered in maple syrup" --workspace "trial_micro_pancake_lambertian" --fp16 --dir_text --albedo --iters 10000 --lambda_depth 3 --h 256 --w 256 --H 800 --W 800 --lambda_rgb 0.1 --min_percent 0.2 --lambda_depthreg 0.1 --anneal_depthreg_weight --default_polar 60 --min_lr 1e-3 --denoising_steps 4 --lambda_orient 1e-2 --lambda_monotonic 0.1 --eta 1.0 --use_scaler --microfacet
# python main.py --text "A stack of pancakes covered in maple syrup" --workspace "trial_micro_pancake_iter=20000" --fp16 --dir_text --albedo --iters 20000 --lambda_depth 3 --h 256 --w 256 --H 800 --W 800 --lambda_rgb 0.1 --min_percent 0.2 --lambda_depthreg 0.1 --anneal_depthreg_weight --default_polar 60 --min_lr 1e-3 --denoising_steps 4 --lambda_orient 1e-2 --lambda_monotonic 0.1 --eta 1.0 --use_scaler --microfacet 
# python main.py --text "A stack of pancakes covered in maple syrup" --workspace "trial_micro_pancake_iter=20000_zvar=10(annealed up)" --fp16 --dir_text --albedo --iters 20000 --lambda_depth 10 --h 256 --w 256 --H 800 --W 800 --lambda_rgb 0.1 --min_percent 0.2 --lambda_depthreg 0.1 --anneal_depthreg_weight --default_polar 60 --min_lr 1e-3 --denoising_steps 4 --lambda_orient 1e-2 --lambda_monotonic 0.1 --eta 1.0 --use_scaler --microfacet 
# python main.py --text "A stack of pancakes covered in maple syrup" --workspace "trial_micro_pancake_iter=20000_zvar=10(no mask)" --fp16 --dir_text --albedo --iters 20000 --lambda_depth 10 --h 256 --w 256 --H 800 --W 800 --lambda_rgb 0.1 --min_percent 0.2 --lambda_depthreg 0.1 --anneal_depthreg_weight --default_polar 60 --min_lr 1e-3 --denoising_steps 4 --lambda_orient 1e-2 --lambda_monotonic 0.1 --eta 1.0 --use_scaler --microfacet --test
# python main.py --text "A stack of pancakes covered in maple syrup" --workspace "trial_micro_pancake_iter=20000_orient=0" --fp16 --dir_text --albedo --iters 20000 --lambda_depth 3 --h 256 --w 256 --H 800 --W 800 --lambda_rgb 0.1 --min_percent 0.2 --lambda_depthreg 0.1 --anneal_depthreg_weight --default_polar 60 --min_lr 1e-3 --denoising_steps 4 --lambda_orient 0.0 --lambda_monotonic 0.1 --eta 1.0 --use_scaler --microfacet
# new zvar implementation with soft mask
# python main.py --text "A stack of pancakes covered in maple syrup" --workspace "trial_micro_pancake_iter=20000_zvar=10_mono=1.0" --fp16 --dir_text --albedo --iters 20000 --lambda_depth 10 --h 256 --w 256 --H 512 --W 512 --lambda_rgb 0.1 --min_percent 0.2 --lambda_depthreg 0.1 --anneal_depthreg_weight --default_polar 60 --min_lr 1e-3 --denoising_steps 4 --lambda_orient 1e-2 --lambda_monotonic 1.0 --eta 1.0 --use_scaler --microfacet --test

python main.py --text "A stack of pancakes covered in maple syrup" --workspace "/home/joseph/Dropbox/4090_backup/stable-dreamfusion-main/trial_ulti_dress" --fp16 --dir_text --albedo --num_steps 128 --upsample_steps 64 --test 
exit
python main.py --text "A stack of pancakes covered in maple syrup" --workspace "/home/joseph/Desktop/dreamfusion_demos/trial_vanilla_pancake_zvar=3_mono=1.0_iters=10000_entropy=1e-2_entropysum=1e-4_orient=0_uniformcdf" --fp16 --dir_text --albedo --num_steps 64 --upsample_steps 32 --test --default_polar 60
# generate some human faces
# maybe: try making the latent move back and forth by adding noise and subtracting noise, with low guidance

# increase batch size maybe
# need ablation study for:
- [] removing depth supervision; (superman/theo example)
- [] removing rgb; (maybe some prompt that specifies color)
- [x] number of steps; (theo statue)
- [] random timestepping
- [] depth variance loss(theo example for thin surfaces)
- [] using the pancake prompt, turn off direction input during test and see what happens
rsync -av -e "ssh -p 22042 -L 8080:localhost:8080 -i ~/.ssh/gcloud" --info=progress2 root@73.190.69.218:~/CS231A/stable-dreamfusion-main ~/4090_backup --exclude="df_ep*.pth" --exclude=trial_450/

rsync -a --prune-empty-dirs -av --include '*/' --include '*.mp4' --exclude '*' josef@ampere1:/lfs/ampere1/0/josef/HiFA_dirty/ .
rsync -a --prune-empty-dirs -av  --include '*/' --include 'trial_vanilla*/**/*.mp4' --exclude '*' josef@ampere1:/lfs/ampere1/0/josef/HiFA_dirty/ .

rsync -arz -v --progress --rsh=ssh -e 'ssh -i /afs/cs.stanford.edu/u/josef/.ssh/gcloud -p 22100 -o StrictHostKeyChecking=no' vastai_kaalia@73.190.69.218::6287257//root/CS231A/stable-dreamfusion-main /lfs/ampere1/0/josef/HiFA_dirty

rsync -arz --prune-empty-dirs -av -e "ssh -p 50447 -L 8080:localhost:8080 -i ~/.ssh/gcloud" --include '*/' --include 'trial_perp**/*.mp4' --exclude '*' root@68.203.212.149:/root/HiFA_dirty/ .

find . -name "*df_ep*.pth" -delete

- [] predict full BRDF(add input & output ray direction to NN input) in addition to albedo(which only takes xyz as coordinate), and shade with a single random direction light source(add input & output ray direction to NN input)
- [] bigger batch size
- [] remove one computation of density outputs
- [] maybe do textureless rendering
- [x] fix metallic to 0 and see what happens
- [] disable orientation loss
- [] randomized resolution training
- [] increase iterations
- [] try cosine annealing timestep