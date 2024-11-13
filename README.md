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
```bash
python train.py --config-name=ssl_methods/barlow_rl.yaml -m
```

Train with the data being available like an ImageFolder.
```bash
python train.py --config-name=ssl_methods/barlow_rl.yaml -m
```

Train for testing SSL method works with CIFAR10.
```bash
python train.py --config-name=ssl_methods/barlow_cifar10.yaml -m
```