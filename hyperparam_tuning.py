# hyperparam_tuning.py

import os
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from env.supply_chain_gym import SupplyChainGymEnv

def sample_ppo_params(trial):
    """
    Suggest a set of PPO hyperparameters.
    """
    return {
        "n_steps": trial.suggest_categorical("n_steps", [64, 128, 256, 512]),
        "gamma": trial.suggest_float("gamma", 0.90, 0.9999, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "ent_coef": trial.suggest_float("ent_coef", 1e-8, 1e-2, log=True),
    }

def objective(trial):
    """
    Train a PPO agent for a small number of timesteps
    and return its mean eval reward.
    """
    # Create a fresh env for training
    train_env = DummyVecEnv([lambda: SupplyChainGymEnv({"max_time": 50})])
    # Sample hyperparameters
    params = sample_ppo_params(trial)
    # Train
    model = PPO("MlpPolicy", train_env, verbose=0, **params)
    model.learn(total_timesteps=50_000)
    # Evaluate
    eval_env = SupplyChainGymEnv({"max_time": 50})
    mean_reward, _ = evaluate_policy(
        model, eval_env, n_eval_episodes=5, deterministic=True
    )
    return mean_reward

if __name__ == "__main__":
    # Ensure optuna logs don’t clutter your workspace
    os.makedirs("optuna_study", exist_ok=True)

    # Run the study
    study = optuna.create_study(
        direction="maximize",
        study_name="ppo_supply_chain_tuning",
        storage=f"sqlite:///optuna_study/ppo.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=20, n_jobs=1)

    print("👉 Best hyperparameters found:")
    for k, v in study.best_params.items():
        print(f"   {k}: {v:.5g}")
    print(f"👉 Best mean reward: {study.best_value:.2f}")

