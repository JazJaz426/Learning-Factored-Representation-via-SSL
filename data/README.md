# CustomDataset class information

`CustomDataset` is a standard PyTorch Dataset class that has access to `__len__()` and `__getitem__()` functions. Below are the inputs and outputs of the `__init__()` and `__getitem__()` functions

## __init__()
Inputs: 
- `data_env_config`: an environment configuration yaml file that can be found in `configs/data_generator/config.yaml` or a copy of the same with specific alterations
- `limits`: maximum number of samples, given that this is a Dataset class for a GymEnv there is no predefined max number of samples
- `policy_model`: policy to be used when sampling the state space for data. Can be `None` if using controlled or random mode generation
- `model_path`: path to the zip file for said model
- `mode`: the mode for data generation (one of few types i.e. `seq` for sequential data as part of rollout, `cont` for controlled data where each factor is randomly sampled, `triplet` for data outputting a triplet of states with one starting state (`s`) and 2 following states (`s1` and `s2`) occuring from different actions, `rand` for repeated random resets)

##__getitem__
Inputs:
- `index`: inbuilt PyTorch indexing for Dataset class

Outputs
- `obs`: data observation, which takes a form based on the `observation_space` value in the `data_env_config` file specified to initialize the class
- `state`: expert feature state that accounts for the underlying factors in the environment


# How to use dataset.py?

There are 2 ways to use the file to generate data

## (1) Building and saving a dataset separately 

Here you would create an instance of `CustomDataset` and use it to collect a bunch of example data, which you will save to images in a folder. Image saving is not handled within `CustomDataset` so would need separate scripts.

```
from PIL import Image

dataset = CustomDataset('configs/data_generator/config.yaml', limit=30, mode='cont')

base_path = os.path.relpath('SOMETHING HERE')

#this will iterate limit number of times
for (obs, label) in dataset:

    img = Image.fromarray(obs)
    img.save(os.path.join(base_path, f'{i+1}.png'))

```

## (2) Creating a dataloader

Here you would use an instance of `CustomDataset` to generate a standard PyTorch `DataLoader` class, which can be used to generate batchified data.

```

dataset = CustomDataset('configs/data_generator/config.yaml', limit=1000, mode='cont')
data_loader = DataLoader(dataset, batch_size=50, shuffle=True)

# this will iterate num batch number of times but limit should be sufficiently big
for batch_idx, (inputs, labels) in enumerate(data_loader):
    print(f"Batch {batch_idx+1}")
    print("Inputs:", inputs.shape)
    print("Labels:", labels.shape) 
```