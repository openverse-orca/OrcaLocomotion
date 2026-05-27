# Orca RL

Orca RL packages a small set of Unitree locomotion demos for OrcaLab playback.
Training follows the bundled Unitree/mjlab project; this repository focuses on
loading those checkpoints and playing them in OrcaLab with the matching scene
assets.

## Supported Tasks

| Task | Training | OrcaLab playback |
| --- | --- | --- |
| `Unitree-Go2-Flat` | Unitree/mjlab | `./play_go2_flat.sh` |
| `Unitree-Go2-Rough` | Unitree/mjlab | `./play_go2_rough.sh` |
| `Unitree-G1-Flat` | Unitree/mjlab | `./play_g1_flat.sh` |

The rough Go2 playback publishes the OrcaLab terrain asset:

```text
assets/001d46537b9e555b/mjlabrough5x5xml_v2/prefabs/terrain_usda
```

The terrain is spawned at `z=0.05`, and the Go2 actor is spawned at the same
height so it starts on the terrain instead of dropping onto it.

## Installation

Ubuntu 22.04 with an NVIDIA GPU is recommended for training. A display is only
needed when playing policies in OrcaLab.

```bash
git clone https://github.com/BenHuHuan/orca_rl.git
cd orca_rl

conda create -n orca_rl python=3.11
conda activate orca_rl

pip install -r requirements.txt
```

For native training setup details, see the bundled Unitree/mjlab guide:

```text
third_party/unitree_rl_mjlab/doc/setup_en.md
third_party/unitree_rl_mjlab/doc/setup_zh.md
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

Start OrcaLab, then run one of the simple playback scripts from the repository
root:

```bash
./play_go2_flat.sh
./play_go2_rough.sh
./play_g1_flat.sh
```

The repository includes smoke-test checkpoints:

```text
checkpoints/test_model_Go2_mjlab_Flat.pt
checkpoints/test_model_Go2_mjlab_Rough.pt
checkpoints/test_model_G1_mjlab_Flat.pt
```

To play a freshly trained checkpoint, pass it as the first argument:

```bash
./play_go2_flat.sh third_party/unitree_rl_mjlab/logs/rsl_rl/go2_velocity/<run>/model_<iter>.pt
./play_go2_rough.sh third_party/unitree_rl_mjlab/logs/rsl_rl/go2_velocity/<run>/model_<iter>.pt
./play_g1_flat.sh third_party/unitree_rl_mjlab/logs/rsl_rl/g1_velocity/<run>/model_<iter>.pt
```

## Assets

Runtime assets kept in this repository are intentionally small:

- debug arrow assets for playback visualization
- primitive XML terrain used by local MuJoCo smoke tests
- mjlab rough 5x5 XML collision terrain and upload zip
- legacy multi-terrain collision XML/hfield still used by local tests

The rough terrain XML package is also available as:

```text
assets/terrain/MjlabRough5x5Xml.zip
```

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
