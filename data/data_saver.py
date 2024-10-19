import pickle
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import SubprocVecEnv
import gymnasium as gym
import argparse

def env_func(name, kwargs):
    return gym.make(name, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--n_envs", type=int, default=32)
    parser.add_argument("--buffer_size", type=int, default=1e6)
    
    args = parser.parse_args()
    
    vec_env = SubprocVecEnv([lambda: env_func(args) for _ in range(args.n_envs)])
    buffer = ReplayBuffer(
        buffer_size=int(args.buffer_size),
        observation_space=vec_env.observation_space, 
        action_space=vec_env.action_space,
        n_envs=args.n_envs,
    )
    
    while buffer.size() < args.buffer_size:
        obs = vec_env.reset()
        actions = vec_env.action_space.sample()
        next_obs, rewards, dones, terminateds, infos = vec_env.step(actions)
        buffer.add(obs, actions, rewards, next_obs, dones, infos)
        obs = next_obs
        for i in range(args.n_envs):
            if dones[i] :
                print(f"Episode {i} finished with reward {rewards[i]}")
                obs[i] = vec_env.reset(i)


    with open("data/buffer.pkl", "wb") as f:
        pickle.dump(buffer, f)
