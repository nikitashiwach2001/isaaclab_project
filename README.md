# TurtleBot Straight Navigation

Train a TurtleBot to move straight forward using PPO in Isaac Lab.

## Setup

1. Install [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)

2. Install this package:
```bash
git clone https://github.com/nikitashiwach2001/isaaclab_project.git
python -m pip install -e source/go_straight
```

3. Verify installation:
```bash
python scripts/list_envs.py
```

## Train

```bash
python scripts/rsl_rl/train.py --task=Isaac-Navigation-TurtlebotStraight-v0
```

Options:
- `--num_envs N` - parallel environments (default: 32)
- `--max_iterations N` - training iterations (default: 600)
- `--video` - record training videos

## Play

```bash
python scripts/rsl_rl/play.py --task=Isaac-Navigation-TurtlebotStraight-v0
```

Options:
- `--num_envs N` - parallel environments
- `--video` - record playback
- `--checkpoint <checkpoint_path>` - load specific checkpoint

Checkpoints are saved to `logs/rsl_rl/turtlebot_straight/`. Trained policies are exported to `logs/rsl_rl/turtlebot_straight/<run>/exported/` as `policy.pt` (JIT) and `policy.onnx`.
