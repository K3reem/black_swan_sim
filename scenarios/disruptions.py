import random

class DisruptionManager:
    def __init__(self, env, sc_env):
        self.env = env
        self.sc_env = sc_env

    def random_disruption(self):
        while True:
            yield self.env.timeout(random.randint(10, 20))
            disruption_type = random.choice([
                "supplier_delay",
                "factory_strike",
                "delivery_block",
            ])

            print(f"⚠️ Black Swan Event: {disruption_type} at time {self.env.now}")

            if disruption_type == "supplier_delay":
                self.sc_env.production_time += 2

            elif disruption_type == "factory_strike":
                # Safely lower the factory's capacity by 1 (min 1)
                # We patch the private _capacity attribute on the SimPy Resource
                old_cap = getattr(self.sc_env.factory, "_capacity", None)
                if old_cap is None:
                    # fallback: use public capacity if it exists
                    old_cap = getattr(self.sc_env.factory, "capacity", 1)
                new_cap = max(1, old_cap - 1)
                setattr(self.sc_env.factory, "_capacity", new_cap)
                print(f"   → Factory capacity now {new_cap}")

            elif disruption_type == "delivery_block":
                self.sc_env.delivery_time += 2
