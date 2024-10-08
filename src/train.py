import torch


def train(agent, env, n_timesteps):
    """
    Train the DQN agent using the built-in learn function of stable-baselines3.
    """
    agent.learn(total_timesteps=n_timesteps)

    print(f"Training complete for {n_timesteps} timesteps.")


def train_run(agent, env, replay_buffer, batch_size, n_episodes, update_freq, do_visualize):
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.predict(state, deterministic=False)[0]
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if do_visualize:
                # Render environment
                env.render()

            # If replay buffer has enough data, sample and train
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                agent.train_on_batch(states, actions, rewards, next_states, dones)
        
        # Update agent policy periodically
        if episode % update_freq == 0:
            agent.update_policy()
        
        print(f"Episode {episode + 1}/{n_episodes}, Reward: {episode_reward}")

