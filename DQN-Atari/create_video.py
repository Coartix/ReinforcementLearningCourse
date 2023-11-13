from gym.wrappers.monitoring.video_recorder import VideoRecorder
import os
import numpy as np
import torch
from torch import nn

def create_video(env, model, device, folder="videos", name="video"):
    video_recorder = VideoRecorder(env, path=os.path.join(folder, name + ".mp4"))
    env.reset()
    s, _, done, _, _ = env.step(0)
    epoch = 0
    total_reward = 0

    while not done and epoch < 1000:
        video_recorder.capture_frame()
        s = np.array(s)
        action_probs = model(torch.tensor(s).unsqueeze(0).to(device))
        action = torch.argmax(action_probs).item()
        s, reward, done, _, _ = env.step(action)
        total_reward += reward
        epoch += 1

    video_recorder.close()
    return total_reward