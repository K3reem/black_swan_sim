\# Supply Chain Black Swan Simulator



An RL-based simulator for supply chain management under black-swan disruptions.

Powered by \*\*SimPy\*\*, \*\*Stable-Baselines3\*\*, \*\*Optuna\*\*, and \*\*Gymnasium\*\*.



!\[Supply Chain RL Simulation](plots\_final/reward.png)



---



\## Project Structure



```

black\_swan\_sim/

├── env/

│   ├── \_\_init\_\_.py

│   ├── supply\_chain\_env.py        # Core SimPy environment

│   └── supply\_chain\_gym.py        # Gym wrapper

├── scenarios/

│   └── disruptions.py             # DisruptionManager logic

├── action\_logging\_cb.py           # TensorBoard action-logging callback

├── train\_sb3\_tb.py                # PPO training with callbacks

├── hyperparam\_tuning.py           # Optuna hyper-parameter tuning

├── train\_best.py                  # Final training with best params

├── evaluate\_and\_plot.py           # Per-episode evaluation \& plots

├── evaluate\_scenarios.py          # Stress-test across disruption rates

├── plots/                         # Generated per-episode charts

├── plots\_final/                   # Final model evaluation charts

├── plots\_scenarios/               # Robustness curve and CSV summary

└── README.md                      # This file

```



---



\## 1. Installation



```bash

git clone https://github.com/<your-username>/black\_swan\_sim.git

cd black\_swan\_sim

python -m venv .venv

\# Activate:

\#   Linux/macOS: source .venv/bin/activate

\#   Windows:     .\\.venv\\Scripts\\activate

pip install -r requirements.txt

```



---



\## 2. Training Pipeline



\### 2.1 Initial Training



```bash

python train\_sb3\_tb.py

```



\* Trains PPO for 100k steps with action-logging and periodic evaluation.

\* Logs to `logs/` and saves best checkpoint in `logs/best\_model/`.



\### 2.2 Hyper-Parameter Tuning



```bash

python hyperparam\_tuning.py

```



\* Runs Optuna for 20 trials over 50k steps each.

\* Stores study in `optuna\_study/`.

\* Prints best parameters.



\### 2.3 Final Training with Best Params



```bash

python train\_best.py

```



\* Retrains PPO for 100k steps using tuned hyper-parameters.

\* Logs to `logs/final\_model/` and saves final model as `ppo\_supply\_chain\_final.zip`.



---



\## 3. Evaluation



\### 3.1 Per-Episode Plots



```bash

python evaluate\_and\_plot.py

```



\* Generates stock, production time, delivery time, and reward plots into `plots\_final/`.



\### 3.2 Robustness Testing



```bash

python evaluate\_scenarios.py

```



\* Runs evaluation at multiple disruption rates (0.0–1.0).

\* Outputs `plots\_scenarios/results\_summary.csv` and `mean\_reward\_vs\_rate.png`.



---



\## 4. Results Summary



📈 \*\*Mean Rewards vs. Disruption Rate\*\*

See `plots\_scenarios/mean\_reward\_vs\_rate.png`



📊 \*\*Numeric Summary\*\* (`results\_summary.csv`)



```csv

\# disruption\_rate, mean\_reward, std\_reward

0.0, 2303.0, 19.89

0.25, 2309.2, 21.62

0.5, 2309.9, 19.86

0.75, 2316.6, 15.85

1.0, 2306.2, 20.96

```



---



\## 5. Next Steps \& Extensions



\* Experiment with different reward functions or multi-agent setups

\* Add more sophisticated disruption scenarios (e.g. cascading or correlated)

\* Dockerize for easier reproducibility and deployment



---



\## 6. License \& Acknowledgements



This project uses:



\* \[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)

\* \[SimPy](https://simpy.readthedocs.io)

\* \[Optuna](https://optuna.org/)

\* \[Gymnasium](https://github.com/Farama-Foundation/Gymnasium)



See `requirements.txt` for full dependencies.



