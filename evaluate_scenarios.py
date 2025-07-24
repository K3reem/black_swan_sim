# evaluate_scenarios.py

import os
import numpy as np
import pandas as pd                         # ← added pandas import
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from env.supply_chain_gym import SupplyChainGymEnv

def stress_test(model_path, rates, n_eval=10, max_time=50):
    """Run evaluate_policy across different disruption rates."""
    model = PPO.load(model_path)
    means, stds = [], []
    for r in rates:
        env = SupplyChainGymEnv({"max_time": max_time, "disruption_rate": r})
        mean, std = evaluate_policy(
            model, env, n_eval_episodes=n_eval, deterministic=True
        )
        print(f"Rate={r:.2f} → mean_reward={mean:.1f} ±{std:.1f}")
        means.append(mean)
        stds.append(std)
    return np.array(means), np.array(stds)

if __name__ == "__main__":
    MODEL = "ppo_supply_chain_final.zip"
    RATES = [0.0, 0.25, 0.5, 0.75, 1.0]

    # 1) Run the stress test
    means, stds = stress_test(MODEL, RATES)

    # 2) Save a numeric summary to CSV
    os.makedirs("plots_scenarios", exist_ok=True)
    df = pd.DataFrame({
        "disruption_rate": RATES,
        "mean_reward":     means,
        "std_reward":      stds
    })
    df.to_csv("plots_scenarios/results_summary.csv", index=False)
    print("Saved numeric summary to plots_scenarios/results_summary.csv")

    # 3) Plot mean vs rate
    plt.errorbar(RATES, means, yerr=stds, fmt='-o')
    plt.title("Mean Reward vs. Disruption Rate")
    plt.xlabel("Disruption Rate")
    plt.ylabel("Mean Eval Reward")
    plt.grid(True)
    plt.savefig("plots_scenarios/mean_reward_vs_rate.png")
    print("✅ Saved summary plot to plots_scenarios/mean_reward_vs_rate.png")

