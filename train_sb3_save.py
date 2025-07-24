from stable_baselines3 import PPO
from env.supply_chain_gym import SupplyChainGymEnv

# 1) Create the env and model
env = SupplyChainGymEnv()
model = PPO("MlpPolicy", env, verbose=1)  

# 2) Train longer under shaped reward
model.learn(total_timesteps=50_000)          # ↑ bumped up

# 3) Save for fast reload later
model.save("ppo_supply_chain")
print("✅ Training complete, model saved to ppo_supply_chain.zip")

