import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env.supply_chain_gym import SupplyChainGymEnv

# Load trained model
model = PPO.load("ppo_supply_chain")

env = SupplyChainGymEnv()
NUM_EP = 3
MAX_STEPS = 100

# Storage
all_times       = []
all_actions     = []
all_warehouses  = []
all_prod_times  = []
all_deliv_times = []

for ep in range(NUM_EP):
    obs = env.reset()
    times, actions, whs, pts, dts = [], [], [], [], []

    for step in range(MAX_STEPS):
        times.append(env.simpy_env.now)
        whs.append(env.sc_env.warehouse.level)
        pts.append(env.sc_env.production_time)
        dts.append(env.sc_env.delivery_time)

        action, _ = model.predict(obs, deterministic=True)
        actions.append(action)

        obs, reward, done, _ = env.step(action)
        if done:
            break

    all_times.append(times)
    all_actions.append(actions)
    all_warehouses.append(whs)
    all_prod_times.append(pts)
    all_deliv_times.append(dts)

# Plot actions alongside warehouse
for i in range(NUM_EP):
    plt.figure(figsize=(8,3))
    plt.step(all_times[i], all_actions[i], where="post")
    plt.title(f"Episode {i+1} Actions Over Time")
    plt.xlabel("SimPy Time")
    plt.ylabel("Action (0=NoOp,1=Prod↑,2=Deliv↑)")
    plt.ylim(-0.2,2.2)
    plt.tight_layout()
    plt.savefig(f"ts_actions_ep{i+1}.png")
    print(f"✅ Saved ts_actions_ep{i+1}.png")
