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
Train with the data generator giving data in an "online" manner.

If you want to run on local machine or in an interact session:
```bash
echo $USER
python train.py --config-name=ssl_methods/barlow_rl.yaml user@_global_=$USER/run_slurm
```

If you want to run as a sbatch script sent to SLURM:
```bash
echo $USER
python train.py --config-name=ssl_methods/barlow_rl.yaml user@_global_=$USER/run_slurm -m
```

Train for testing SSL method works with CIFAR10.
```bash
echo $USER
python train.py --config-name=ssl_methods/barlow_cifar10.yaml user@_global_=$USER/run_slurm
```

Run method 1
```bash
echo $USER
python train.py --config-name=ssl_methods/covariance_factor.yaml user@_global_=$USER/run_slurm
```

Run method 2
```bash
echo $USER
python train.py --config-name=ssl_methods/mask_factor.yaml user@_global_=$USER/run_slurm
```