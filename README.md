# Learning-Factored-Representation-via-SSL
CSCI2952X SSL Project


# File Structure

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

# SSL Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

# run ssl training script
```bash
python models/ssl_models/train.py --config-name=ssl_methods/barlow_cifar10_multigpu.yaml ++/user@_global_: vipul/run_slurm
```