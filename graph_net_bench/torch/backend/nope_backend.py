import torch
import torch_musa
from .graph_compiler_backend import GraphCompilerBackend


class NopeBackend(GraphCompilerBackend):
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, model):
        return model

    def synchronize(self):
        if torch.musa.is_available():
            torch.musa.synchronize()
