from abc import abstractmethod
import numpy as np

class Scheduler:
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer  
        self.step_count = 0         

    @abstractmethod
    def step(self):
        pass

class StepLR(Scheduler):
    def __init__(self, optimizer, step_size=30, gamma=0.1) -> None:
        super().__init__(optimizer)
        self.step_size = step_size 
        self.gamma = gamma         

    def step(self) -> None:
        self.step_count += 1
        if self.step_count >= self.step_size:
            self.optimizer.lr *= self.gamma 
            self.step_count = 0  

class MultiStepLR(Scheduler):
    """Multi-interval scheduler: Decays the learning rate at specified milestone steps"""
    def __init__(self, optimizer, milestones=[30, 60, 90], gamma=0.1) -> None:
        super().__init__(optimizer)
        self.milestones = sorted(milestones)  
        self.gamma = gamma
        self.current_idx = 0  # the index of the current milestone 

    def step(self) -> None:
        self.step_count += 1
        while self.current_idx < len(self.milestones) and self.step_count >= self.milestones[self.current_idx]:
            self.optimizer.lr *= self.gamma  
            self.current_idx += 1  # move to the next milestone

class ExponentialLR(Scheduler):
    """Exponential decay scheduler: Decays the learning rate by an exponential factor at each step"""
    def __init__(self, optimizer, gamma=0.99) -> None:
        super().__init__(optimizer)
        self.gamma = gamma  

    def step(self) -> None:
        self.step_count += 1
        self.optimizer.lr *= self.gamma 