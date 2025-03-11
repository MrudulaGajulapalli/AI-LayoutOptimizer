import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from room_env import FurnitureArrangementEnv

# Load the trained model
try:
    model = PPO.load("furniture_model.zip")
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading model:", e)
    exit()

# Initialize environment
env = FurnitureArrangementEnv()
obs = env.reset()

# Get model prediction
action, _ = model.predict(obs)

# Take a step in the environment
obs, reward, done, _ = env.step(action)

# Get furniture positions directly from the environment
furniture_positions = obs.reshape(env.num_furniture, 2)

#  2D Visualization using Matplotlib
def plot_layout(room_size, furniture_positions, furniture_sizes):
    fig, ax = plt.subplots(figsize=(6, 6))  # Adjust figure size
    ax.set_xlim(0, room_size[0])
    ax.set_ylim(0, room_size[1])
    ax.set_xticks(range(room_size[0] + 1))
    ax.set_yticks(range(room_size[1] + 1))
    ax.set_title("Optimized Furniture Layout", fontsize=14)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Draw room boundary
    ax.plot([0, room_size[0], room_size[0], 0, 0], 
            [0, 0, room_size[1], room_size[1], 0], "k-", linewidth=2)

    # Draw furniture
    colors = ["blue", "green", "red", "purple", "orange"]  # Different colors for clarity
    for idx, ((x, y), (w, h)) in enumerate(zip(furniture_positions, furniture_sizes)):
        ax.add_patch(plt.Rectangle((x, y), w, h, color=colors[idx % len(colors)], alpha=0.5))
        ax.text(x + w / 2, y + h / 2, f"F{idx+1}", color="white", ha="center", va="center", fontsize=12)

    plt.show()

# Display the structured layout
plot_layout(env.room_size, furniture_positions, env.furniture_sizes)
