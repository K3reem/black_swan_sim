import simpy
import random

class SupplyChainEnv:
    def __init__(self, env, disruption_rate: float = 1.0, render_mode=None):
        self.render_mode = render_mode
        self.env = env
        self.disruption_rate = disruption_rate
        self.supplier = simpy.Resource(env, capacity=1)
        self.factory = simpy.Resource(env, capacity=1)
        self.warehouse = simpy.Container(env, capacity=100, init=50)
        self.production_time = 5
        self.delivery_time = 3

    def source_materials(self):
        yield self.env.timeout(random.randint(2, 4))  # supplier delay

    def manufacture(self):
        yield self.env.timeout(self.production_time)

    def deliver(self):
        yield self.env.timeout(self.delivery_time)

    def run_process(self, name):
        while True:
            with self.supplier.request() as req:
                yield req
                print(f"{name} sourcing at {self.env.now}")
                yield self.env.process(self.source_materials())

            with self.factory.request() as req:
                yield req
                print(f"{name} manufacturing at {self.env.now}")
                yield self.env.process(self.manufacture())

            yield self.env.process(self.deliver())
            self.warehouse.put(1)
            print(f"{name} delivered to warehouse at {self.env.now}")

from scenarios.disruptions import DisruptionManager

def main():
    env = simpy.Environment()
    sc_env = SupplyChainEnv(env)
    disruption_mgr = DisruptionManager(env, sc_env)

    for i in range(3):
        env.process(sc_env.run_process(f"Order_{i+1}"))

    env.process(disruption_mgr.random_disruption())

    env.run(until=50)

if __name__ == "__main__":
    main()
