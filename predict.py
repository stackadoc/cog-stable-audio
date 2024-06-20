# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import json
from os.path import abspath, dirname

import torch
import torchaudio
from cog import BasePredictor, Input, Path
from einops import rearrange

from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict
from weights_downloader import WeightsDownloader

MODEL_PATH = "/src/models"
WEIGHTS_STR = "stable-audio-open-1.0"


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model, self.model_config = self._load_model()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(device=self.device)

    def _load_model(
        self,
    ):
        weights_downloader = WeightsDownloader()
        weights_downloader.download_weights(WEIGHTS_STR, MODEL_PATH)
        model_config_path = f"{MODEL_PATH}/{WEIGHTS_STR}/model_config.json"
        model_ckpt_path = f"{MODEL_PATH}/{WEIGHTS_STR}/model.ckpt"
        with open(model_config_path) as f:
            model_config = json.load(f)
        model = create_model_from_config(model_config)
        model.load_state_dict(load_ckpt_state_dict(model_ckpt_path))
        return model, model_config

    def predict(
        self,
        prompt: str = Input(),
        negative_prompt: str = Input(default=""),
        seconds_start: int = Input(default=0),
        seconds_total: int = Input(default=8, le=47),
        cfg_scale: float = Input(default=6.0),
        steps: int = Input(default=100),
        seed: int = Input(default=-1),
        sampler_type: str = Input(default="dpmpp-3m-sde"),
        sigma_min: float = Input(default=0.03),
        sigma_max: int = Input(default=500),
        init_noise_level: float = Input(default=1.0),
        batch_size: int = Input(default=1),
    ) -> Path:
        if not self.model or not self.model_config:
            self.model, self.model_config = self._load_model()

        sample_rate = self.model_config["sample_rate"]
        sample_size = sample_rate * (seconds_total + 1)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Prompt: {prompt}")

        conditioning = [
            {
                "prompt": prompt,
                "seconds_start": seconds_start,
                "seconds_total": seconds_total,
            }
        ] * batch_size

        if negative_prompt:
            negative_conditioning = [
                {
                    "prompt": negative_prompt,
                    "seconds_start": seconds_start,
                    "seconds_total": seconds_total,
                }
            ] * batch_size
        else:
            negative_conditioning = None

        seed = int(seed)

        audio = generate_diffusion_cond(
            self.model,
            conditioning=conditioning,
            negative_conditioning=negative_conditioning,
            steps=steps,
            cfg_scale=cfg_scale,
            batch_size=batch_size,
            sample_size=sample_size,
            sample_rate=sample_rate,
            seed=seed,
            device=self.device,
            sampler_type=sampler_type,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            init_noise_level=init_noise_level,
        )

        audio = rearrange(audio, "b d n -> d (b n)")

        audio = (
            audio.to(torch.float32)
            .div(torch.max(torch.abs(audio)))
            .clamp(-1, 1)
            .mul(32767)
            .to(torch.int16)
            .cpu()
        )

        wav_path = "output.wav"
        torchaudio.save(wav_path, audio, sample_rate)

        return Path(wav_path)
