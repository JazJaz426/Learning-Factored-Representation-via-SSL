# Learning-Factored-Representation-via-SSL
Project objectives:
* Decouple state representation and policy learning.
* Learn disentangled representation.
* Ablation studies for data diet.
* Training dynamics analysis to see ***how*** disentangled representations emerge.

This project was made for "CSCI2952X-Research Topics in Self Supervised Learning" at Brown University.

# Code Map - SSL
* Access the model training and user-specific configs here: `configs/ssl_configs/ssl_methods` and `configs/ssl_configs/user/$USER`.
* The custom dataset wrapper for RL rollout + SSL: `data/ssl_dataset`.
* The evaluation metrics for embeddings: `disentanglement_metrics`.
* The proposed model architecture code: `models/ssl_models`.
* The main trainer function: `train.py`.

# File Structure - RL

```
data \
  data_loader.py \
  data_generator.py \
  data_saver.py \
  src \
    observations \
    data.csv \
models \
  TED \
    ...py files... \
  expert \
  visual \
  model1 \
  model2 \
  policy_head \
    ...ppo and dqn... \
configs \
  VISSL \
    ... yaml file ... \
  models \
    ... yaml file ... \
  data_generator \
    ... yaml file ... \
evals \
  policy \
  disentanglement \
  interpretability \
tests \
  PyTorch_vision_tests \
```

# SSL Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

> **Caution**: You will need to create user specific configs for $USER run command to work. Create them here: `configs/ssl_configs/user/$USER`.

# run ssl training script
```bash
source .venv/bin/activate
echo $USER
```

Train with the data generator which does rollout but doesn't update policy, i.e., offline RL.

If you want to run on local machine or in an interact session:
```bash
python train.py --config-name=ssl_methods/barlow_rl.yaml user@_global_=$USER/run_slurm
```

If you want to run as a sbatch script sent to SLURM:
```bash
python train.py --config-name=ssl_methods/barlow_rl.yaml user@_global_=$USER/run_slurm -m
```

Run method 1
```bash
python train.py --config-name=ssl_methods/covariance_factor.yaml user@_global_=$USER/run_slurm -m
```

Run method 2
```bash
python train.py --config-name=ssl_methods/mask_factor.yaml user@_global_=$USER/run_slurm -m
```

Run eval only for debugging
```bash
python train.py --config-name=ssl_methods/barlow_rl_eval.yaml user@_global_=$USER/run_slurm -m
```

Run train+eval for debugging
```bash
python train.py --config-name=ssl_methods/barlow_rl.yaml user@_global_=$USER/run_debug -m
```

# Interpreting the labels for the RL dataset:
* The state or the normalized state information is label.
* Example: 
```bash
{'agent_dir': np.int64(1), 'agent_pos': (np.int64(4), np.int64(6)), 'door_locked': 0, 'door_open': 0, 'door_pos': (np.int64(2), np.int64(3)), 'goal_pos': (np.int64(3), np.int64(3)), 'holding_key': 0, 'key_pos': (np.int64(4), np.int64(4))} 
[np.float64(0.3333333333333333), np.float64(0.5), np.float64(0.75), np.float64(0.0), np.float64(0.0), np.float64(0.25), np.float64(0.375), np.float64(0.375), np.float64(0.375), np.float64(0.0), np.float64(0.5), np.float64(0.5)]
```
* Each value in the state is normalized by [value - min(0 here)/[max(8 here) - min(0 here)]]
