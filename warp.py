from transformers.generation.logits_process import LogitsWarper
import torch.nn.functional as F
import torch
from logzero import logger


class GPUTemperatureLogitsWarper(LogitsWarper):
    def __init__(
        self,
        device_id: int = 0,
        gpu_temp_min: int = 20,
        gpu_temp_max: int = 100,
        inf_temp_min: int = 0,
        inf_temp_max: int = 5,
    ):
        if (not torch.cuda.is_available()) or (device_id > torch.cuda.device_count()):
            raise ValueError(f"Cannot find cuda device {device_id}")

        self.device_id = device_id
        self.gpu_temp_min = gpu_temp_min
        self.gpu_temp_max = gpu_temp_max
        self.inf_temp_min = inf_temp_min
        self.inf_temp_max = inf_temp_max

    def _get_temp_from_temp(self):
        temp = torch.cuda.temperature(device=self.device_id)
        logger.debug(f"Using GPU temperature: {temp}")
        return (
            (temp - self.gpu_temp_min) / (self.gpu_temp_max - self.gpu_temp_min)
        ) * (self.inf_temp_max - self.inf_temp_min)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        temperature = self._get_temp_from_temp()
        logger.debug(f"Using inference temperature: {temperature}")
        scores = scores / temperature
        return scores
