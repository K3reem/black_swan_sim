# train_sb3_tb.py

import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

# import your new Gym-wrapped environment
from env.supply_chain_gym import SupplyChainGymEnv
from action_logging_cb import ActionLoggingCallback

def main():
    # ensure log folders exist
    os.makedirs("logs/best_model", exist_ok=True)
    os.makedirs("logs/eval_logs", exist_ok=True)

    # build vectorized train & eval envs (each gets its own SimPy sim)
    train_env = DummyVecEnv([lambda: SupplyChainGymEnv({"max_time": 50})])
    eval_env  = DummyVecEnv([lambda: SupplyChainGymEnv({"max_time": 50})])

    # create PPO agent with TensorBoard logging
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        tensorboard_log="logs/"
    )

    # callbacks: action-logging + periodic evaluation
    action_cb = ActionLoggingCallback()
    eval_cb   = EvalCallback(
        eval_env,
        best_model_save_path="logs/best_model/",
        log_path="logs/eval_logs/",
        eval_freq=10_000,
        deterministic=True,
        render=False
    )

    # kick off training
    model.learn(
        total_timesteps=100_000,
        callback=[action_cb, eval_cb]
    )

    # save the final policy
    model.save("ppo_supply_chain")
    print("✅ Done. Model saved to ppo_supply_chain.zip")

if __name__ == "__main__":
    main()

