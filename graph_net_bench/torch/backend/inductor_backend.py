import torch
from .graph_compiler_backend import GraphCompilerBackend

# Supported modes: "default" and "max_autotune"
INDUCTOR_MODE_MAP = {
    "default": "default",
    "max_autotune": "max-autotune",
    "reduce_overhead": "reduce-overhead",
}


class InductorBackend(GraphCompilerBackend):
    def __init__(self, config):
        super().__init__(config)
        self.mode = self._resolve_mode(config)

    @staticmethod
    def _resolve_mode(config):
        mode = config.get("mode", "default")
        if mode not in INDUCTOR_MODE_MAP:
            raise ValueError(
                f"Unknown inductor mode: {mode!r}. "
                f"Supported modes: {list(INDUCTOR_MODE_MAP.keys())}"
            )
        return INDUCTOR_MODE_MAP[mode]

    def __call__(self, model):
        return torch.compile(model, backend="inductor", mode=self.mode)

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
