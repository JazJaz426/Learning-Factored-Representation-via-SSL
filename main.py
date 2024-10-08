from src.gym import SimpleEnv

import csv
import cv2
import numpy as np


def save_video(imgs, filename):
    height, width, layers = imgs[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    video = cv2.VideoWriter(filename, fourcc, 10, (width, height)) 

    for img in imgs:
        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video.write(bgr_img)  

    video.release()  


def main():
    env = SimpleEnv(render_mode="rgb_array")

    # List to hold (s, a, s') triplets
    data = []

    num_episodes = 10  # Number of episodes to run
    for episode in range(num_episodes):
        obs = env.reset()  # Reset the environment for a new episode
        state = {"pos": (int(env.agent_pos[0]),int(env.agent_pos[1])), "dir": env.agent_dir}
        state_list = [state]
        done = False

        img = env.render()
        imgs = [img]
        
        for i in range(20):
            action = env.action_space.sample()  # Sample a random action
            next_obs, reward, done, info, _ = env.step(action)  # Take a step in the environment
            
            # Record (s, a, s') triplet
            state = {"pos": (int(env.agent_pos[0]),int(env.agent_pos[1])), "dir": env.agent_dir}
            data.append((state_list[-1], action, state))
            state_list.append(state)

            
            # Optionally render the environment
            img = env.render()
            imgs.append(img)

        # Save imgs as MP4 video for the current episode
        video_filename = f'episode_{episode + 1}.mp4'
        save_video(imgs, video_filename)

    # Save the data to a CSV file
    with open('agent_snapshots.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['State', 'Action', 'Next State'])  # Header
        for s, a, s_prime in data:
            writer.writerow([s, a, s_prime])


    env.close()


    
if __name__ == "__main__":
    main()