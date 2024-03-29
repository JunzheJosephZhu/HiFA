<h2><p align="center">
 HiFA: High-fidelity Text-to-3D Generation with Advanced Diffusion Guidance<br>
</p></h2>
<h3><p align="center">
  <a href="https://arxiv.org/abs/2305.18766">paper</a> | <a href="https://josephzhu.com/HiFA-site/">project page</a>
</p></h3>

<div align="center">
  <img src="https://github.com/JunzheJosephZhu/HiFA-site/blob/master/HiFA_files/gifs_v3/bunny_rgb_normal.mp4.gif" width="660"/><br>
  <b>Text-to-3d</b>
</div>
<div align="center">
  <img src="https://github.com/JunzheJosephZhu/HiFA-site/blob/master/HiFA_files/image_guided_v3/dinosaur3.jpeg" width="330"/>
  <img src="https://github.com/JunzheJosephZhu/HiFA-site/blob/master/HiFA_files/image_guided_v3/dinosaur_df_ep0100_rgb.gif" width="330"/><br>
  <b>Image-guided 3d generation</b>
</div>
<div align="center">
  <img src="https://github.com/JunzheJosephZhu/HiFA-site/blob/master/HiFA_files/image_to_3d_v3/teapot.png" width="330"/>
  <img src="https://github.com/JunzheJosephZhu/HiFA-site/blob/master/HiFA_files/image_to_3d_v3/teapot_df_ep0100_rgb.gif" width="330"/><br>
  <b>Image-to-3d reconstruction</b>
</div>


### Install
```
conda create -n hifa python=3.9
pip install -r requirements.txt
(Suppose your are using torch2.0 + cu117, install torch-scatter:)
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
make sure torch is with cuda.is_available()
```
### Build extension (optional)
By default, we use [`load`](https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load) to build the extension at runtime.
We also provide the `setup.py` to build each extension:
```bash
# install all extension modules
bash scripts/install_ext.sh

# if you want to install manually, here is an example:
pip install ./raymarching # install to python path (you still need the raymarching/ folder, since this only installs the built extension.)
```
This is a lot more convenient, since you wont have to rebuild it

### Example commands

#### Text to 3d
```
CUDA_VISIBLE_DEVICES=0 python main.py --text "a baby bunny sitting on top of a stack of pancakes" --workspace trials_throne_sanity --dir_text --albedo --phi_range 0 120 
```
#### Image to 3d reconstruction / Image-guided 3d generation
For both of those, you need to generate some predicted views following instruction in [SyncDreamer](https://github.com/liuyuan-pal/SyncDreamer#preparation-for-training) by first removing the background and then generating 16 views. Copy over the output 0.png to this project's folder and specify the file with image-path.
We provided some example images under raw_input and gt_images.

After you get the predicted views from SyncDreamer, for image to 3d generation:
```
CUDA_VISIBLE_DEVICES=0 python main.py --text "A toy grabber with dinosaur head" --learned_embeds_path "gt_images/dinosaur/learned_embeds.bin" --image_path "gt_images/dinosaur/0.png" --workspace "trials_dinosaur(textprompt)_imgto3d" --dir_text --albedo --gt_image_rate 0.5 --h 256 --w 256
```
For image-guided 3d generation:
```
CUDA_VISIBLE_DEVICES=0 python main.py --text "A toy grabber with dinosaur head" --learned_embeds_path "gt_images/dinosaur/learned_embeds.bin" --image_path "gt_images/dinosaur/0.png" --workspace "trials_dinosaur(textprompt)_imgguided" --dir_text --albedo --gt_image_rate 0.5 --h 256 --w 256 --anneal_gt 0.7
```
To use textual inversion, first compute token:
```
python textual-inversion/textual_inversion.py --output_dir="gt_images/teapot" --train_data_dir="raw_input/no_bg/teapot"  --initializer_token="teapot"  --placeholder_token="_teapot_placeholder_" --pretrained_model_name_or_path="SG161222/Realistic_Vision_V5.1_noVAE" --learnable_property="object"  --resolution=256 --train_batch_size=1 --gradient_accumulation_steps=4 --max_train_steps=5000 --learning_rate=5.0e-4 --scale_lr --lr_scheduler="constant" --lr_warmup_steps=0 --use_augmentations

python main.py --text "a DSLR photo of <token>" --learned_embeds_path "gt_images/teapot/learned_embeds.bin" --image_path "gt_images/teapot/0.png" --workspace "trials_teapot_gtrate=0.5_v9" --dir_text --albedo --gt_image_rate 0.5 --h 256 --w 256
``````
However, we don't find that textual inversion visibly brings a benefit

#### Note: add --clip_grad option if NaN value is produced during training
----

Some notable additions compared to the paper: 
1. More [kernel smoothing strategies](https://github.com/JunzheJosephZhu/HiFA/blob/be1a1fa42c66ff388255f6f4db8a2fda0309a35d/nerf/renderer.py#L31C23-L37C23) for coarse-to-fine sampling
2. New regularizers akin to z-variance: [monotonicity loss](https://github.com/JunzheJosephZhu/HiFA/blob/69463fbcc1bef23c711ee043960e71198459626f/nerf/utils.py#L594C2-L600C3) and [z-entropy loss](https://github.com/JunzheJosephZhu/HiFA/blob/69463fbcc1bef23c711ee043960e71198459626f/nerf/utils.py#L618C1-L625C1)
   Intuitively, monotonicity loss ensures that the blending weight first increases then decreases monotonically along a ray, ensuring that there is a consistent surface. z-entropy loss regulates the entropy of the blending weight distribution along the ray, similar to how z-variance loss regulates its variance.
3. This is just a clarification, since I noticed this issue in threestudio's re-implementation of [z-variance loss](https://github.com/JunzheJosephZhu/HiFA/blob/69463fbcc1bef23c711ee043960e71198459626f/nerf/utils.py#L580-L590): the rendering weights have to be normalized(divided by sum) along each ray before z-variance is computed. This is because if you don't normalize, two things will happen: i) the depth and z-variance will both be zero for background pixels 2) you cannot set lambda_zvar to be big, since then the model will be encouraged to have more background pixels(which have low z-variance), leading to small objects. This issue goes away with normalization.


# Acknowledgement
* This repo is modified from [stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion). We will provide an alternative version on [threestudio](https://github.com/threestudio-project/threestudio) soon, which will also make our contributions fully integrated with [ProlificDreamer](https://ml.cs.tsinghua.edu.cn/prolificdreamer/)

# Citation
```
@misc{zhu2023hifa,
      title={HiFA: High-fidelity Text-to-3D Generation with Advanced Diffusion Guidance}, 
      author={Junzhe Zhu and Peiye Zhuang},
      year={2023},
      eprint={2305.18766},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
