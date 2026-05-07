import numpy as np
from omegaconf import DictConfig


class SamplingController:
    def __init__(self, config: DictConfig, initial_temperature: float):
        self.warmup_steps = config.warmup_steps
        self.target_entropy = config.target_entropy
        self.temperature = initial_temperature
        self.max_temperature = config.temperature.max
        self.min_temperature = config.temperature.min
        self.delta_temperature = config.temperature.delta

    def update(self, metrics: dict, global_steps: int):
        if global_steps <= self.warmup_steps:
            return

        current_entropy = metrics["actor/entropy"]
        if current_entropy < self.target_entropy: 
            self.temperature += self.delta_temperature
        else:
            self.temperature -= self.delta_temperature

        self.temperature = float(np.clip(self.temperature, self.min_temperature, self.max_temperature))

        return

    def get_metrics(self):
        return {
            "sampling_control/temperature": self.temperature,
        }