# Don't Throw Away the Solution: Training-Free Acceleration of Flow-Based Robot Policies

<div align="center">
  <a href="./README.md"><img src="https://img.shields.io/badge/README-English-111111.svg"></a>
  <a href="./README_zh.md"><img src="https://img.shields.io/badge/README-%E4%B8%AD%E6%96%87-d14836.svg"></a>
  <a href="https://github.com/H-EmbodVis/RePIC/blob/main/docs/static/pdfs/RePIC.pdf"><img src="https://img.shields.io/badge/Paper-PDF-b31b1b?logo=readthedocs&logoColor=white"></a>
  <a href="https://h-embodvis.github.io/RePIC/"><img src="https://img.shields.io/badge/Homepage-project-orange.svg?logo=googlehome"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square"></a>

  <h5 align="center"><em><a href="https://github.com/HyperbolicCurve">Xianjin Wu</a><sup>1,*</sup>, <a href="https://github.com/lixian-lipton">Xian Li</a><sup>1,*</sup>, Jianxiang Tang<sup>1</sup>, <a href="https://dk-liang.github.io/">Dingkang Liang</a><sup>1</sup>, <a href="https://scholar.google.com/citations?user=UeltiQ4AAAAJ&hl=en">Xiang Bai</a><sup>1</sup></em></h5>
  <sup>1</sup> Huazhong University of Science and Technology, <sup>*</sup> Equal contribution
</div>

<div align="center">
  <a href="docs/static/images/intro.png">
    <img src="docs/static/images/intro.png" alt="RePIC overview" width="100%">
  </a>
</div>

This repository contains the official implementation of **RePIC** (the paper above): a training-free sampler that reuses the discarded solver trajectory of the previous replan to start the next one, and refines it with batched fixed-point updates. It is built on [Fast-WAM](https://github.com/yuantianyuan01/FastWAM) (`main` branch) and also ships implementations for π0.5 (`pi`) and GR00T N1.5 (`gr00t`).

## 📄 Abstract

Flow-based robot policies generate each action chunk by sequential denoising and discard the solver trajectory at every replan, even though consecutive control cycles solve closely related problems. We introduce **RePIC**, a training-free sampler that shifts this cached trajectory to the new horizon, re-aligns it to fresh noise with a closed-form correction, and refines it with batched fixed-point updates of the pretrained network, turning an accurate initialization into fewer sequential network calls. On Fast-WAM, RePIC reduces action-generation latency by **2.66×** on RoboTwin 2.0 and **2.75×** on LIBERO at comparable success (**5.71×** on RoboTwin 2.0 with RePIC-Flash), accelerates π0.5 and GR00T N1.5 by 1.14× and 1.68× without retraining, and raises real-robot success from 66.3% to 77.5%.

## 🔍 Method

<div align="center">
  <a href="docs/static/images/pipeline.png">
    <img src="docs/static/images/pipeline.png" alt="RePIC pipeline" width="100%">
  </a>
</div>

## 📣 News

- `2026.10`: Released the paper and the code for Fast-WAM (`main`), π0.5 (`pi`) and GR00T N1.5 (`gr00t`).

## 📈 Performance

Latency is the action-generation time per chunk on an NVIDIA RTX 4090 (48 GB), excluding observation capture, simulator stepping and action execution. The tables and the figure below are taken from the paper.

<div align="center">
  <a href="docs/static/images/perf_robotwin.png">
    <img src="docs/static/images/perf_robotwin.png" alt="Paper Table 1: results on RoboTwin 2.0" width="100%">
  </a>
</div>

<div align="center">
  <a href="docs/static/images/perf_libero.png">
    <img src="docs/static/images/perf_libero.png" alt="Paper Table 2: results on LIBERO" width="100%">
  </a>
</div>

<div align="center">
  <a href="docs/static/images/real_robot.png">
    <img src="docs/static/images/real_robot.png" alt="Paper Fig. 5: real-world deployment" width="100%">
  </a>
</div>

## 🌿 Repository branches

The repository has three self-contained implementation branches:

| Branch | Base project | RePIC scope |
| --- | --- | --- |
| `main` | [Fast-WAM](https://github.com/yuantianyuan01/FastWAM) | LIBERO and RoboTwin 2.0; original sampler, RePIC, and benchmark-specific Flash backends |
| `pi` | [OpenPI](https://github.com/Physical-Intelligence/openpi) π0.5 | LIBERO only; RePIC without a separate Flash implementation |
| `gr00t` | [GR00T N1.5](https://github.com/NVIDIA/Isaac-GR00T) | LIBERO only; RePIC without a separate Flash implementation |

Clone the branch of the implementation you want to use:

```bash
git clone --branch main https://github.com/H-EmbodVis/RePIC.git
# Or: git clone --branch pi https://github.com/H-EmbodVis/RePIC.git
# Or: git clone --branch gr00t https://github.com/H-EmbodVis/RePIC.git
cd RePIC
```

The `pi` and `gr00t` branches follow their upstream projects; their installation and evaluation steps are documented on each branch. The rest of this README describes the `main` branch.

## ⚙️ Installation

```bash
conda create -n repic python=3.10 -y
conda activate repic
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

RePIC-Flash on RoboTwin additionally needs [Torch-TensorRT](https://github.com/pytorch/TensorRT); install the build that matches your PyTorch and CUDA versions (`import torch_tensorrt` must succeed). The LIBERO Flash backend only needs CUDA.

## Dataset/Model Preparation

RePIC is training-free and reuses the released Fast-WAM checkpoints. Model, dataset and checkpoint preparation follows Fast-WAM.

### 1. Wan2.2 backbone and ActionDiT

```bash
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"   # optional, default ./checkpoints

python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

### 2. Released Fast-WAM checkpoints

```bash
pip install -U huggingface_hub

huggingface-cli download yuanty/fastwam \
  libero_uncond_2cam224.pt \
  libero_uncond_2cam224_dataset_stats.json \
  robotwin_uncond_3cam_384.pt \
  robotwin_uncond_3cam_384_dataset_stats.json \
  --local-dir ./checkpoints/fastwam_release
```

### 3. Benchmarks

**LIBERO.** Install the official [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) environment, then `pip install mujoco==3.3.2` (keep it consistent with the LIBERO data version). The preprocessed training dataset is not needed for evaluation.

**RoboTwin 2.0.** The evaluation code of [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) is vendored in `third_party/RoboTwin`. Follow the official instructions to finish the environment and asset setup, then link the policy:

```bash
ln -sfn "$(pwd)/experiments/robotwin/fastwam_policy" "$(pwd)/third_party/RoboTwin/policy/fastwam_policy"
```

## 🚀 Evaluation

`repic`, `repic_flash` and `repic_config` select the sampler at evaluation time; no checkpoint changes are needed:

| `repic` | `repic_flash` | Runtime path |
| --- | --- | --- |
| `false` | `false` | Original Fast-WAM sampler |
| `false` | `true` | Original sampler with the Flash backend |
| `true` | `false` | RePIC (eager PyTorch) |
| `true` | `true` | **RePIC-Flash** |

The Flash backend is CUDA Graph replay on LIBERO and a TensorRT FP16 action core with CUDA Graph replay on RoboTwin (batches padded to `B=10`). It is not initialized when `repic_flash=false`, and an unavailable backend raises an error instead of silently falling back to eager execution.

> **Reproducing the paper.** The released Fast-WAM checkpoints were trained with action scheduler shift `5.0`, while the default is now `1.0`, so pass `EVALUATION.sigma_shift=5.0` for every command below. `EVALUATION.compile_action_infer` (default `true` on LIBERO) only speeds up the plain original sampler (`repic=false`, `repic_flash=false`); the RePIC and Flash paths do not use it, so set it to `false` if you want an eager baseline for latency comparisons.

### LIBERO

```bash
CKPT=./checkpoints/fastwam_release/libero_uncond_2cam224.pt
STATS=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json

# Original sampler
python experiments/libero/run_libero_manager.py task=libero_uncond_2cam224_1e-4 \
  ckpt=$CKPT EVALUATION.dataset_stats_path=$STATS EVALUATION.sigma_shift=5.0 \
  EVALUATION.repic=false MULTIRUN.num_gpus=8

# RePIC
python experiments/libero/run_libero_manager.py task=libero_uncond_2cam224_1e-4 \
  ckpt=$CKPT EVALUATION.dataset_stats_path=$STATS EVALUATION.sigma_shift=5.0 \
  EVALUATION.repic=true EVALUATION.repic_config=configs/repic/libero.json \
  MULTIRUN.num_gpus=8

# RePIC-Flash (CUDA Graph)
python experiments/libero/run_libero_manager.py task=libero_uncond_2cam224_1e-4 \
  ckpt=$CKPT EVALUATION.dataset_stats_path=$STATS EVALUATION.sigma_shift=5.0 \
  EVALUATION.repic=true EVALUATION.repic_flash=true \
  EVALUATION.repic_config=configs/repic/libero.json MULTIRUN.num_gpus=8
```

### RoboTwin 2.0

```bash
CKPT=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt
STATS=./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json

# Original sampler
python experiments/robotwin/run_robotwin_manager.py task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=$CKPT EVALUATION.dataset_stats_path=$STATS EVALUATION.sigma_shift=5.0 \
  EVALUATION.repic=false MULTIRUN.num_gpus=8

# RePIC (with residual-guided commitment, R_n in {16, ..., 24})
python experiments/robotwin/run_robotwin_manager.py task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=$CKPT EVALUATION.dataset_stats_path=$STATS EVALUATION.sigma_shift=5.0 \
  EVALUATION.repic=true EVALUATION.repic_config=configs/repic/robotwin.json \
  MULTIRUN.num_gpus=8

# RePIC-Flash (TensorRT FP16 + CUDA Graph)
python experiments/robotwin/run_robotwin_manager.py task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=$CKPT EVALUATION.dataset_stats_path=$STATS EVALUATION.sigma_shift=5.0 \
  EVALUATION.repic=true EVALUATION.repic_flash=true \
  EVALUATION.repic_config=configs/repic/robotwin.json MULTIRUN.num_gpus=8
```

The managers default to 8 GPUs; pass a smaller `MULTIRUN.num_gpus` if needed. The RoboTwin default is `EVALUATION.task_config=demo_randomized`; use `EVALUATION.task_config=demo_clean` for the clean setting. We evaluate with **unseen** instructions (`EVALUATION.instruction_type=unseen`), following Motus. `EVALUATION.skip_get_obs_within_replan=true` (default) skips RGB rendering inside an action chunk, which makes saved videos look low-FPS; set it to `false` for fully rendered videos.

### RePIC parameters

The paper settings live in `configs/repic/libero.json` and `configs/repic/robotwin.json` (the `_comment` fields document each key):

| Key | LIBERO | RoboTwin | Meaning |
| --- | ---: | ---: | --- |
| `num_inference_steps` | 10 | 10 | integration steps `S` of the original sampler |
| `picard_window` | 10 | 10 | window `p` of parallel fixed-point updates |
| `picard_tol` | 0.05 | 0.05 | stride tolerance `τ` (threshold `τ²σ²`) |
| `stride_residual` | `prefix` | `prefix` | residual scope of Eq. 4: executed prefix `1:R_n` (`prefix`) or the whole chunk (`full`, ablation) |
| `r_max` | 10 | 24 | execution length of the original sampler, `R_max` |
| `dynamic_replan_enabled` | `false` | `true` | residual-guided commitment; requires `R_max > ⌈H/2⌉` |
| `dynamic_replan_log_mean` / `_std` | −3.5 / 0.5 | −3.5 / 0.5 | log-normal commitment map `μ`, `ς` |

`replan_steps` must equal `r_max` (the defaults in `configs/sim_*.yaml` already do), and the LIBERO action ensembler must stay disabled. The action horizon is `H=32` on both benchmarks.

## 🧪 Tests

CPU-only unit tests check the sampler against plain serial Euler sampling, the noise-aligned transfer (Eq. 6–7), the prefix-versus-full stride test, and the RoboTwin replan logic:

```bash
pip install pytest
pytest tests -q
```

## 📁 File Structure

```text
RePIC/
├── configs/
│   ├── data/                 # Dataset configs (LIBERO, RoboTwin)
│   ├── model/                # Model architecture configs
│   ├── repic/                # RePIC parameters (libero.json, robotwin.json)
│   └── task/                 # Task-level configs
├── experiments/
│   ├── libero/               # LIBERO evaluation (run_libero_manager.py)
│   └── robotwin/             # RoboTwin evaluation (run_robotwin_manager.py, fastwam_policy/)
├── src/fastwam/models/wan22/
│   ├── fastwam.py            # RePIC sampler: Picard loop, noise-aligned transfer, Flash backends
│   ├── schedulers/           # Flow-matching scheduler and Picard integration
│   └── tensorrt_action.py    # TensorRT-lowerable action core (RoboTwin Flash)
├── scripts/                  # Training and preprocessing entrypoints
├── tests/                    # CPU unit tests
├── third_party/RoboTwin/     # Vendored RoboTwin evaluation code
└── docs/                     # Project homepage (GitHub Pages)
```

## 👍 Acknowledgement

RePIC builds on the following projects:

* [Fast-WAM](https://github.com/yuantianyuan01/FastWAM): the world action model, training code and checkpoints this codebase is derived from.
* [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) and [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO): the evaluation benchmarks.
* [OpenPI](https://github.com/Physical-Intelligence/openpi) (π0.5) and [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) (GR00T N1.5): the VLA backbones of the `pi` and `gr00t` branches.
* [Wan2.2](https://github.com/Wan-Video/Wan2.2): the video backbone of Fast-WAM.
* The parallel sampling formulation follows [ParaDiGMS](https://github.com/AndyShih12/paradigms) (Shih et al., 2023).

## 📖 Citation

If you find this repository useful in your research, please consider giving a star ⭐ and a citation.

```bibtex
@misc{wu2026repic,
  title={Don't Throw Away the Solution: Training-Free Acceleration of Flow-Based Robot Policies},
  author={Xianjin Wu and Xian Li and Jianxiang Tang and Dingkang Liang and Xiang Bai},
  year={2026},
  note={Preprint}
}
```

Please also cite [Fast-WAM](https://arxiv.org/abs/2603.16666), on which the `main` branch is built.

## License

This project is released under the MIT License (see [LICENSE](./LICENSE)). It is derived from Fast-WAM, © The FastWAM Authors; the `pi` and `gr00t` branches inherit the licenses of their upstream projects.
