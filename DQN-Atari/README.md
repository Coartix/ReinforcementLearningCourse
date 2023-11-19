Hugo Deplagne

## DQN-Atari Project from DeepMind paper [Human-level control through deep reinforcement learning](./nature14236.pdf)  

#### This project is a part of the Reinforcement Learning course at EPITA (2023).  

This project is using gym and torch modules to train a Deep Q-Network on the Atari Breakout game.  

You'll find the implementation inside the `torch_DQL.ipynb` notebook.  
The parameters chosen for the training are the ones from the paper, except for the memory size that had to be reduced to fit in the RAM of my small computer 😞
Also another preprocess has been done to the images, we use luminance instead of grayscale.    

The training can take many hours to complete depending on your hardware and the number of episodes you want to train on.  

The DQN network can be found inside the `DQN_torch.py`, the memory inside `ReplayBuffer.py` and the video function inside `create_video.py`.  

The notebook contains simple and straightforward usage of the functions.  

#### What has been done:

- The memory size had to be reduced to fit in the RAM of my small computer, to remediate some of this loss we use uint8 variables inside ReplayBuffer instead of float32. This quantization reduces considerably the memory size. So no scaling (no normalization) is done on the images.  

- Another configuration of AtariPreprocessing has been done in `preprocess.py`, it adds a luminance_obs boolean parameter to use luminance instead of grayscale.

- The training time is very long, even with a GPU  

- The training is not very stable, as it sometimes reduces the rewards  

- Many repositories replicate the paper's result using Tensorflow, this one does it in Pytorch  

- A video function has been added to visualize the result of the agent after training  

#### Results:

With my computer, I was able to train the agent on thousands of episodes with memory size of 15 000. This resulted as far as a mean reward of 14.  

On another computer with enough RAM to hold a memory size of 200 000, I had resulted after about 6000 episodes to a mean reward of 27.  

#### Next:

I'm looking forward to make a model playing Breakout based on a Decision Transformer.

