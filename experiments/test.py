import traci
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from experiments.Infra import Config_SUMO, Infra, SECTION_RESULT, TOTAL_RESULT
from experiments.RunSimulation import RunSimulation
from experiments.inframanager import InfraManager
import os
from sumo_rl import SumoEnvironment

class CustomSumoEnvironment(SumoEnvironment):
    def __init__(self, simInfra, **kwargs):
        super().__init__(**kwargs)
        self._cust_step = 0
        self._cust_infra:Infra = simInfra

    def _sumo_step(self):
        self.sumo.simulationStep()
        self._cust_infra.update()
        self._cust_step += 1
        totalr = TOTAL_RESULT.TOTAL_CO2_ACC.name
        print('step = ', self._cust_step,' TOTAL CO2:', self._cust_infra.getTotalCO2mg())



# 콜백 클래스를 정의하여 학습 결과를 매번 출력하도록 설정
class EveryStepCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(EveryStepCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        if self.verbose > 0:
            print(f"Step: {self.num_timesteps}, Reward: {self.locals['rewards'][-1]}")
        return True

class LearningManager(RunSimulation):
    def __init__(self):
        super().__init__(Config_SUMO, 'RL_DQL', isExternalSignal=True)
        print('L init1')
        self.model = DQN.load("dqn_model_episode_72.zip")
        self.env = CustomSumoEnvironment(
            net_file=self.config.scenario_file,
            single_agent=True,
            route_file=self.config.route_file,
            use_gui=True,
            yellow_time=4,
            min_green=5,
            max_green=120,
            sumo_seed=1,
            simInfra=self.getInfra()
        )
        print('L init2')

    def preinit(self):
        pass

    def run_simulation(self):
        obs, _ = self.env.reset()
        done = False
        total_reward = 0
        step = 0
        maxstep = 11700 / self.env.delta_time

        while step <=maxstep:
            action, _states = self.model.predict(obs,state=None, deterministic=False)
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            step += 1

        print(f"Total CO2 Emission Reward: {total_reward}")

runner = LearningManager()
runner.run_simulation()