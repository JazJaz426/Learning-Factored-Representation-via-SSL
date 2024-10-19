from stable_baselines3.common.env_checker import check_env
from data_generator import DataGenerator

env = DataGenerator()
check_env(env)