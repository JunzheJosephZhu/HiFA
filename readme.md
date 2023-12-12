<h2><p align="center">
 HiFA: High-fidelity Text-to-3D Generation with Advanced Diffusion Guidance<br>
</p></h2>
<h3><p align="center">
  <a href="https://arxiv.org/abs/2305.18766">paper</a> | <a href="https://hifa-team.github.io/HiFA-site/">project page</a>
</p></h3>

<div align="center">
  <img src="https://github.com/HiFA-team/HiFA-site/blob/master/HiFA_files/gifs_v3/bunny_rgb_normal.mp4.gif" width="660"/><br>
  <b>Text-to-3d</b>
</div>
<div align="center">
  <img src="https://github.com/HiFA-team/HiFA-site/blob/master/HiFA_files/image_guided_v3/dinosaur3.jpeg" width="330"/>
  <img src="https://github.com/HiFA-team/HiFA-site/blob/master/HiFA_files/image_guided_v3/dinosaur_df_ep0100_rgb.gif" width="330"/><br>
  <b>Image-guided 3d generation</b>
</div>
<div align="center">
  <img src="https://github.com/HiFA-team/HiFA-site/blob/master/HiFA_files/image_to_3d_v3/teapot.png" width="330"/>
  <img src="https://github.com/HiFA-team/HiFA-site/blob/master/HiFA_files/image_to_3d_v3/teapot_df_ep0100_rgb.gif" width="330"/><br>
  <b>Image-to-3d reconstruction</b>
</div>


### Install
```
pip install -r requirements.txt

make sure torch is with cuda.is_available()
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

(Suppose your are using torch2.0 + cu 117, install torch-scatter:)
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
pip install torchmetrics
pip install invisible_watermark transformers accelerate safetensors

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

### Example commands

#### Text to 3d
```
CUDA_VISIBLE_DEVICES=0 python main.py --text "iron throne from game of thrones" --workspace trials_throne_sanity --dir_text --albedo --phi_range 0 120 
```
#### Image to 3d reconstruction / Image-guided 3d generation
For both of those, need to generate some predicted views from [SyncDreamer](https://github.com/liuyuan-pal/SyncDreamer), and specify the file with --image_path
Then for image to 3d generation:
```
CUDA_VISIBLE_DEVICES=0 python main.py --text "A toy grabber with dinosaur head" --learned_embeds_path "gt_images/dinosaur/learned_embeds.bin" --image_path "gt_images/dinosaur/0.png" --workspace "trials_dinosaur(textprompt)_imgto3d_sanity" --dir_text --albedo --min_percent 0.3  --dir_rate 0.5 --gt_image_rate 0.5 --h 256 --w 256
```
For image-guided 3d generation:
```
CUDA_VISIBLE_DEVICES=0 python main.py --text "A toy grabber with dinosaur head" --learned_embeds_path "gt_images/dinosaur/learned_embeds.bin" --image_path "gt_images/dinosaur/0.png" --workspace "trials_dinosaur(textprompt)_imgguided_sanity" --dir_text --albedo --min_percent 0.3  --dir_rate 0.5 --gt_image_rate 0.5 --h 256 --w 256 --anneal_gt 0.7
```
#### Note: add --clip_grad option if NaN value is produced during training
----

Some notable additions compared to the paper: 
1. More kernel smoothing strategies for coarse-to-fine sampling
2. New regularizers akin to z-variance: [monotonicity loss](https://github.com/KelestZ/HIFA_dirty/blob/287240e812c2e0eddd6c646d400e8b802c132b32/nerf/utils.py#L592C1-L593C1) and [z-entropy loss](https://github.com/KelestZ/HIFA_dirty/blob/287240e812c2e0eddd6c646d400e8b802c132b32/nerf/utils.py#L618C1-L619C1)
   Intuitively, monotonicity loss ensures that the blending weight first increases then decreases monotonically along a ray, ensuring that there is a consistent surface. Z-entropy loss regulates the entropy of the blending weight distribution along the ray, similar to how z-variance loss regulates its variance.



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
