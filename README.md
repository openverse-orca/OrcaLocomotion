# Orca RL

Orca RL packages a small set of Unitree locomotion demos for OrcaLab playback.
Training follows the bundled Unitree/mjlab project; this repository focuses on
loading those checkpoints and playing them in OrcaLab with the matching scene
assets.

## Supported Tasks

| Task | Training | OrcaLab playback |
| --- | --- | --- |
| `Unitree-Go2-Flat` | Unitree/mjlab | `orca_rl.run_play` |
| `Unitree-Go2-Rough` | Unitree/mjlab | `orca_rl.run_play` |
| `Unitree-G1-Flat` | Unitree/mjlab | `orca_rl.run_play` |

## Installation

Ubuntu 22.04 with an NVIDIA GPU is recommended for training. A display is only
needed when playing policies in OrcaLab.

```bash
git clone https://github.com/BenHuHuan/orca_rl.git
cd orca_rl

# System dependencies for building bundled C++ extensions
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev

conda create -n orca_rl python=3.11
conda activate orca_rl

pip install -r requirements.txt
```

## Training

Train with the bundled Unitree/mjlab project. These commands run the native
mjlab trainer headlessly and write checkpoints under
`third_party/unitree_rl_mjlab/logs/rsl_rl/`.

```bash
cd third_party/unitree_rl_mjlab
python scripts/train.py Unitree-Go2-Flat --env.scene.num-envs=4096
python scripts/train.py Unitree-Go2-Rough --env.scene.num-envs=4096
python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096
```

The resulting checkpoints can be passed to Orca RL playback.

## Playback

Start OrcaLab, then run the playback CLI from the repository root:

```bash
python -m orca_rl.run_play \
  --config Unitree-Go2-Flat \
  --policy-backend mjlab \
  --checkpoint <checkpoint.pt>

python -m orca_rl.run_play \
  --config Unitree-Go2-Rough \
  --policy-backend mjlab \
  --checkpoint <checkpoint.pt>

python -m orca_rl.run_play \
  --config Unitree-G1-Flat \
  --policy-backend mjlab \
  --checkpoint <checkpoint.pt>
```

Use a checkpoint produced by the Unitree/mjlab training step.

## Assets

Upload the XML terrain packages to OrcaLab before using terrain playback:

| Asset package | Use |
| --- | --- |
| `assets/terrain/OrcaPrimitiveTerrainXml.zip` | Primitive XML terrain smoke tests |
| `assets/terrain/MjlabRough5x5Xml.zip` | Go2 rough terrain playback |

Upload them in OrcaLab's XML asset upload flow. The rough Go2 playback uses the
uploaded rough terrain asset and spawns it at `z=0.05`; the Go2 actor starts at
the same height so it begins on the terrain instead of dropping onto it.

## Development

Useful checks before publishing changes:

```bash
python -m compileall orca_rl assets/terrain/mjlab_rough_5x5_xml/generate_terrain.py
git diff --check
```

## Acknowledgments

Training tasks and robot assets are based on the bundled
`third_party/unitree_rl_mjlab` project, which builds on
[mujocolab/mjlab](https://github.com/mujocolab/mjlab).
