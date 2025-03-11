import gym
from stable_baselines3 import PPO
from room_env import FurnitureArrangementEnv

# Ensure Gym is registered correctly
try:
    env = FurnitureArrangementEnv()
    print("Environment initialized successfully!")
except Exception as e:
    print("Error initializing environment:", e)
    exit()

# Train the model
try:
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("furniture_model.zip")  # Explicitly save as .zip
    print("Training complete. Model saved as 'furniture_model.zip'.")
except Exception as e:
    print("Error during training:", e)
