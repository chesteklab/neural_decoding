from abc import ABC, abstractmethod

class decoder:
    @abstractmethod
    def __init__(self, input_size, output_size, model_params) -> None:
        pass

    # todo add call which just calls forward
    
    @abstractmethod
    def train(self, input, output):
        pass

    @abstractmethod
    def forward(self, input):
        pass

    @abstractmethod
    def save_model(self, filepath):
        pass
    
    @abstractmethod
    def load_model(self, filepath):
        pass