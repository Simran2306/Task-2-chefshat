import gym
from gym import spaces
import numpy as np
import os


class ChefsHatEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(ChefsHatEnv, self).__init__()

        self.episodeNumber = 0
        self.logDirectory = "./logs"
        self.verbose = False
        self.saveLog = False

        os.makedirs(self.logDirectory, exist_ok=True)

        # ✅ ACTIONS: 0–4
        self.action_space = spaces.Discrete(5)

        # 🔥 NEW: MEANINGFUL STATE (NOT 1D)
        # [player_score, opponent_score, step, last_action]
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(4,), dtype=np.float32
        )

        self.max_steps = 20

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.episodeNumber += 1
        self.current_step = 0

        # 🔥 GAME STATE
        self.player_score = 0
        self.opponent_score = 0
        self.last_action = 0

        return self._get_obs()

    def _get_obs(self):
        return np.array([
            self.player_score,
            self.opponent_score,
            self.current_step,
            self.last_action
        ], dtype=np.float32)

    def step(self, action):
        self.current_step += 1

        # 🔥 opponent plays randomly
        opponent_action = np.random.randint(0, 5)

        # 🔥 scoring logic
        if action > opponent_action:
            reward = 1
            self.player_score += action
        elif action < opponent_action:
            reward = -1
            self.opponent_score += opponent_action
        else:
            reward = 0

        self.last_action = action

        done = self.current_step >= self.max_steps

        # bonus for winning
        if done:
            if self.player_score > self.opponent_score:
                reward += 5
                winner = 0
            else:
                reward -= 5
                winner = 1
        else:
            winner = -1

        return self._get_obs(), float(reward), done, {"winner": winner}

    def render(self, mode="human"):
        print(f"P:{self.player_score} | O:{self.opponent_score}")

    def close(self):
        pass