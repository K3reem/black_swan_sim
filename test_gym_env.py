from env.supply_chain_gym import SupplyChainGymEnv

env = SupplyChainGymEnv()

obs, _ = env.reset()
print(f"Initial observation: {obs}")

for step in range(10):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print(f"Step {step + 1}: Action={action}, Obs={obs}, Reward={reward}")
    if done:
        break
