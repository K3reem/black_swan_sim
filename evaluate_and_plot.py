# evaluate_and_plot.py

import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env.supply_chain_gym import SupplyChainGymEnv

def evaluate_and_collect(model_path, n_episodes=20, max_time=50):
    model    = PPO.load(model_path)
    env      = SupplyChainGymEnv({"max_time": max_time})
    all_stock, all_prod, all_deliv, all_rew = [], [], [], []

    for _ in range(n_episodes):
        obs, _ = env.reset()               # unpack (obs, info)
        done    = False
        s_hist, p_hist, d_hist, r_hist = [], [], [], []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            s_hist.append(obs[0])
            p_hist.append(obs[1])
            d_hist.append(obs[2])
            r_hist.append(reward)
        all_stock.append(s_hist)
        all_prod.append(p_hist)
        all_deliv.append(d_hist)
        all_rew.append(r_hist)

    return all_stock, all_prod, all_deliv, all_rew

def plot_metric(histories, title, ylabel, save_as):
    plt.figure(figsize=(8,4))
    for h in histories:
        plt.plot(h, alpha=0.3)
    plt.title(title)
    plt.xlabel("Time step")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_as)
    print(f"Saved plot: {save_as}")

if __name__ == "__main__":
    best_model    = "ppo_supply_chain_final.zip"
    output_folder = "plots_final"

    stock, prod, deliv, rew = evaluate_and_collect(best_model, n_episodes=20, max_time=50)

    os.makedirs(output_folder, exist_ok=True)
    plot_metric(stock,  "Warehouse Stock over Time",  "Stock Level",  f"{output_folder}/stock.png")
    plot_metric(prod,   "Production Time over Time",   "Prod. Time",    f"{output_folder}/prod_time.png")
    plot_metric(deliv,  "Delivery Time over Time",     "Deliv. Time",   f"{output_folder}/deliv_time.png")
    plot_metric(rew,    "Per-Step Reward over Time",   "Reward",        f"{output_folder}/reward.png")

    print("✅ Evaluation & plotting complete.")

