# train_best.py

import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env.supply_chain_gym import SupplyChainGymEnv

# ← these came straight from your Optuna run
best_params = {
    "n_steps": 128,
    "gamma": 0.97601,
    "learning_rate": 0.0001844,
    "ent_coef": 2.1607e-05,
}

def main():
    # ensure folder for logs exists
    os.makedirs("logs/final_model", exist_ok=True)

    # single vectorized env with max_time=50
    env = DummyVecEnv([lambda: SupplyChainGymEnv({"max_time": 50})])

    # initialize PPO with your best hyperparams
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="logs/final_model/",
        **best_params
    )

    # train full 100k timesteps
    model.learn(total_timesteps=100_000)

    # save the final policy
    model.save("ppo_supply_chain_final")
    print("✅ Final model saved as ppo_supply_chain_final.zip")

if __name__ == "__main__":
    main()

