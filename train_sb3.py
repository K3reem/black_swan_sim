from stable_baselines3 import PPO
from env.supply_chain_gym import SupplyChainGymEnv

# Create the Gym env
env = SupplyChainGymEnv()

# Instantiate the agent
model = PPO("MlpPolicy", env, verbose=1)

# Train for 10k timesteps
model.learn(total_timesteps=10_000)

# Evaluate: run 5 episodes
for ep in range(5):
    obs = env.reset()
    done = False
    total = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        total += reward
    print(f"Episode {ep+1} reward: {total:.1f}")

