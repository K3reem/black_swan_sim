import gymnasium as gym
from gymnasium import spaces
import numpy as np
import simpy
from env.supply_chain_env import SupplyChainEnv
from scenarios.disruptions import DisruptionManager

class SupplyChainGymEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, config=None, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.disruption_rate = config.get("disruption_rate", 1.0) if isinstance(config, dict) else 1.0
        self.simpy_env = simpy.Environment()
        self.sc_env = SupplyChainEnv(self.simpy_env, disruption_rate=self.disruption_rate)

        dm = DisruptionManager(self.simpy_env, self.sc_env)
        self.simpy_env.process(dm.random_disruption())

        self.max_time = config.get("max_time", 50) if isinstance(config, dict) else 50
        self.current_time = 0

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 1.0, 1.0], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.simpy_env = simpy.Environment()
        self.sc_env = SupplyChainEnv(self.simpy_env)
        dm = DisruptionManager(self.simpy_env, self.sc_env)
        self.simpy_env.process(dm.random_disruption())
        self.current_time = 0
        self.simpy_env.run(until=1)
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        prev_prod = self.sc_env.production_time
        prev_deliv = self.sc_env.delivery_time

        if action == 1:
            self.sc_env.production_time = max(1, prev_prod - 1)
        elif action == 2:
            self.sc_env.delivery_time = max(1, prev_deliv - 1)

        self.simpy_env.run(until=self.simpy_env.now + 1)
        self.current_time = self.simpy_env.now

        obs = self._get_obs()

        stock_reward = float(self.sc_env.warehouse.level)
        time_penalty = 1.0 * self.sc_env.production_time + 0.5 * self.sc_env.delivery_time
        prod_redux = max(0.0, prev_prod - self.sc_env.production_time)
        deliv_redux = max(0.0, prev_deliv - self.sc_env.delivery_time)
        bonus = 2.0 * prod_redux + 1.0 * deliv_redux
        reward = stock_reward - time_penalty + bonus

        terminated = self.current_time >= self.max_time
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            print(
                f"Time {self.current_time} | Stock {self.sc_env.warehouse.level} | "
                f"ProdTime {self.sc_env.production_time} | DelivTime {self.sc_env.delivery_time}"
            )

    def _get_obs(self):
        return np.array([
            self.sc_env.warehouse.level,
            self.sc_env.production_time,
            self.sc_env.delivery_time
        ], dtype=np.float32)

