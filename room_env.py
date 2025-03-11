import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt

class FurnitureArrangementEnv(gym.Env):
    def __init__(self, room_size=(10, 10), num_furniture=3):
        super(FurnitureArrangementEnv, self).__init__()

        self.room_size = room_size  # Room dimensions (width, height)
        self.num_furniture = num_furniture  # Number of furniture items

        # Furniture size constraints (width, height)
        self.furniture_sizes = [(2, 2), (3, 2), (2, 3)]  # Random sizes

        # Action space: Move furniture in 4 directions (left, right, up, down)
        self.action_space = spaces.Discrete(4 * num_furniture)

        # Observation space: Positions of all furniture
        # self.observation_space = spaces.Box(
        #     low=0, high=max(room_size), shape=(num_furniture, 2), dtype=np.int32
        # )
        self.observation_space = spaces.Box(
            low=0, high=max(self.room_size), shape=(self.num_furniture * 2,), dtype=np.int32
        )


        self.reset()

    def reset(self):
        # Randomly place furniture in the room
        self.furniture_positions = np.random.randint(
            0, min(self.room_size), (self.num_furniture, 2)
        )
        return self.furniture_positions.flatten()

    def step(self, action):
        """Moves one furniture item based on action"""
        furniture_id = action // 4
        move_direction = action % 4

        dx, dy = 0, 0
        if move_direction == 0:  # Left
            dx = -1
        elif move_direction == 1:  # Right
            dx = 1
        elif move_direction == 2:  # Up
            dy = -1
        elif move_direction == 3:  # Down
            dy = 1

        # Update furniture position within room constraints
        new_x = np.clip(self.furniture_positions[furniture_id][0] + dx, 0, self.room_size[0] - 1)
        new_y = np.clip(self.furniture_positions[furniture_id][1] + dy, 0, self.room_size[1] - 1)
        self.furniture_positions[furniture_id] = [new_x, new_y]

        # Calculate reward
        reward = self.calculate_reward()
        done = False  # No termination condition for now

        return self.furniture_positions.flatten(), reward, done, {}

    def calculate_reward(self):
        """Encourages spacing between furniture and wall constraints"""
        reward = 0

        # Penalize furniture overlap
        for i in range(self.num_furniture):
            for j in range(i + 1, self.num_furniture):
                if np.array_equal(self.furniture_positions[i], self.furniture_positions[j]):
                    reward -= 10  # Heavy penalty for overlap

        return reward

    def render(self):
        """Visualize room and furniture positions"""
        plt.figure(figsize=(5, 5))
        plt.xlim(0, self.room_size[0])
        plt.ylim(0, self.room_size[1])
        for i, (x, y) in enumerate(self.furniture_positions):
            plt.scatter(x, y, s=500, label=f"F{i}")
        plt.legend()
        plt.grid()
        plt.show()
