# action_logging_cb.py

from stable_baselines3.common.callbacks import BaseCallback

class ActionLoggingCallback(BaseCallback):
    """
    Logs each action taken (per env in a VecEnv) to TensorBoard
    under tags action/0, action/1, …
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # SB3 makes the raw actions available in self.locals["actions"]
        actions = self.locals.get("actions")
        if actions is None:
            return True

        # Flatten vectorized actions if needed
        try:
            flat_actions = actions.flatten()
        except AttributeError:
            flat_actions = [actions]

        # Log each action as a separate scalar
        for idx, act in enumerate(flat_actions):
            self.logger.record(f"action/{idx}", float(act))

        return True

