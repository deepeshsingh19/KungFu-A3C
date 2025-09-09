# Kung Fu A3C Agent

An implementation of **Asynchronous Advantage Actor-Critic (A3C)** to train an AI agent on the **Kung Fu Master Atari game** using **PyTorch** and **Gymnasium**.

## Demo Video
Below is a sample run of the trained agent:
[![Image](https://github.com/user-attachments/assets/fd43e04a-2b36-4bf8-8092-b5cb3eecc441)]

## Features

* Preprocessing of Atari frames (resize, grayscale, frame stacking).
* Convolutional neural network for policy + value estimation.
* A3C training loop with multiple environments in parallel.
* Evaluation of trained agent.
* Video recording and playback of the agent’s gameplay.

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

### Install system dependency
On macOS:
```bash
brew install swig
```

## Project Structure

* **Network**: CNN-based actor-critic model.
* **PreprocessAtari**: Preprocessing wrapper for Atari observations.
* **Agent**: A3C implementation (policy + value updates).
* **EnvBatch**: Parallel environment batch for faster training.
* **Training Loop**: Runs agent across environments with progress tracking.
* **Visualization**: Saves and displays gameplay videos.

## ▶Usage

Run the notebook or script:

```bash
python kungfu_a3c.py
```

To train and evaluate:

* Training loop runs for **3000 iterations** (configurable).
* Progress shown via `tqdm`.
* Average rewards logged every 1000 steps.

To visualize trained agent:

```python
show_video_of_model(agent, env)
show_video()
```

## Environment

The agent is trained on **KungFuMasterNoFrameskip-v0** from Gymnasium’s Atari environments.
Action names can be printed with:

```python
print(env.env.env.env.get_action_meanings())
```

## Results

* The agent gradually learns to maximize reward using A3C.
* Videos of gameplay are saved as `video.mp4`.

## Notes

* Make sure `ale-py` is installed; otherwise Atari games won’t load.
* CUDA is automatically used if available.
* You can tweak hyperparameters like learning rate, number of environments, etc.

