from __future__ import annotations

import json
import gc
from pathlib import Path
from time import perf_counter
from typing import Generator

import numpy as np

from .local_modules import REPO_ROOT, ensure_voxcpm_on_path
from .tts_types import TTSRequestOptions, TTSRuntimeConfig


ensure_voxcpm_on_path()

from voxcpm import VoxCPM  # type: ignore  # noqa: E402
from voxcpm.model.voxcpm import LoRAConfig  # type: ignore  # noqa: E402


def _default_lora_config() -> LoRAConfig:
    return LoRAConfig(
        enable_lm=True,
        enable_dit=True,
        enable_proj=False,
        r=32,
        alpha=16,
        target_modules_lm=["q_proj", "v_proj", "k_proj", "o_proj"],
        target_modules_dit=["q_proj", "v_proj", "k_proj", "o_proj"],
    )


class VoxCpmTtsService:
    def __init__(self, config: TTSRuntimeConfig | None = None):
        self.config = config or TTSRuntimeConfig()
        self._model = None
        self._active_lora_path: str | None = None
        self._active_model_version: str | None = None
        self._warmup_signature: tuple[str | None, ...] | None = None
        self._warmup_result: dict[str, float | bool | int | None] | None = None

    @property
    def is_warmed(self) -> bool:
        return self._warmup_result is not None

    _MODEL_DIR_BY_VERSION = {
        "1.5": "openbmb__VoxCPM1.5",
        "2": "openbmb__VoxCPM2",
    }

    @staticmethod
    def _normalize_model_version(model_version: str | None) -> str:
        return "2" if str(model_version or "").strip() == "2" else "1.5"

    def _resolve_model_path(self, model_version: str | None = None) -> str:
        version = self._normalize_model_version(model_version)
        dir_name = self._MODEL_DIR_BY_VERSION[version]

        roots: list[Path] = []
        if self.config.model_path:
            roots.append(Path(self.config.model_path).expanduser().parent)
        roots.append(REPO_ROOT / "assets" / "tts")
        roots.append(REPO_ROOT / "voxcpm-tts-streaming-module" / "models")

        for root in roots:
            candidate = root / dir_name
            if candidate.exists():
                return str(candidate)

        if self.config.model_path and Path(self.config.model_path).exists():
            return self.config.model_path
        raise FileNotFoundError(f"VoxCPM model path not found for version {version}: {dir_name}")

    def _resolve_lora_root(self) -> Path:
        if self.config.lora_root:
            return Path(self.config.lora_root).expanduser().resolve()
        return (REPO_ROOT / "lora").resolve()

    def get_model(self, lora_ready: bool = False, model_version: str | None = None) -> VoxCPM:
        version = self._normalize_model_version(model_version)
        needs_version_switch = (
            self._model is not None and self._active_model_version != version
        )
        needs_lora_reinit = (
            lora_ready
            and self._model is not None
            and getattr(getattr(self._model, "tts_model", None), "lora_config", None) is None
        )
        if needs_version_switch or needs_lora_reinit:
            old_model = self._model
            self._model = None
            self._active_lora_path = None
            self._active_model_version = None
            del old_model
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        if self._model is None:
            model_path = self._resolve_model_path(version)
            self._model = VoxCPM(
                voxcpm_model_path=model_path,
                enable_denoiser=self.config.load_denoiser,
                optimize=self.config.optimize,
                lora_config=_default_lora_config() if lora_ready else None,
            )
            self._active_model_version = version
            if lora_ready:
                self._model.set_lora_enabled(False)
        return self._model

    def list_lora_checkpoints(self) -> list[dict[str, str | None]]:
        root = self._resolve_lora_root()
        root.mkdir(parents=True, exist_ok=True)
        checkpoints: list[dict[str, str | None]] = []
        for checkpoint_dir in root.rglob("*"):
            if not checkpoint_dir.is_dir():
                continue
            if not (checkpoint_dir / "lora_weights.safetensors").is_file() and not (checkpoint_dir / "lora_weights.ckpt").is_file():
                continue
            rel_path = checkpoint_dir.relative_to(root).as_posix()
            base_model = None
            config_path = checkpoint_dir / "lora_config.json"
            if config_path.is_file():
                try:
                    payload = json.loads(config_path.read_text(encoding="utf-8"))
                    base_model = payload.get("base_model")
                except Exception:
                    base_model = None
            checkpoints.append(
                {
                    "path": rel_path,
                    "label": rel_path,
                    "base_model": str(base_model) if base_model else None,
                }
            )
        return sorted(checkpoints, key=lambda item: item["path"] or "", reverse=True)

    def _resolve_lora_selection(self, selection: str | None) -> Path | None:
        raw = str(selection or "").strip()
        if not raw or raw.lower() == "none":
            return None
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = self._resolve_lora_root() / raw
        candidate = candidate.resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"LoRA checkpoint not found: {candidate}")
        return candidate

    def _apply_lora(self, selection: str | None, model_version: str | None = None) -> None:
        checkpoint = self._resolve_lora_selection(selection)
        if checkpoint is None:
            model = self.get_model(lora_ready=False, model_version=model_version)
            if getattr(getattr(model, "tts_model", None), "lora_config", None) is not None:
                model.set_lora_enabled(False)
            self._active_lora_path = None
            return
        model = self.get_model(lora_ready=True, model_version=model_version)
        checkpoint_path = str(checkpoint)
        if self._active_lora_path != checkpoint_path:
            model.load_lora(checkpoint_path)
            self._active_lora_path = checkpoint_path
        model.set_lora_enabled(True)

    def _apply_seed(self, seed: int | None) -> None:
        if seed is None or int(seed) < 0:
            return
        import torch

        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    def sample_rate(self, options: TTSRequestOptions | None = None) -> int:
        request_options = options or TTSRequestOptions()
        return int(
            self.get_model(
                lora_ready=bool(request_options.lora_selection),
                model_version=request_options.model_version,
            ).tts_model.sample_rate
        )

    def _resolve_reference_inputs(
        self,
        options: TTSRequestOptions | None = None,
        prompt_wav_path: str | None = None,
        prompt_text: str | None = None,
    ) -> tuple[str | None, str | None]:
        request_options = options or TTSRequestOptions()
        final_prompt_wav = request_options.prompt_audio_path or prompt_wav_path
        final_prompt_text = request_options.prompt_text or prompt_text

        if bool(final_prompt_wav) != bool(final_prompt_text):
            return None, None

        if not final_prompt_wav:
            return None, None

        candidate = Path(final_prompt_wav).expanduser()
        if not candidate.exists():
            return None, None

        return str(candidate), final_prompt_text

    def _resolve_reference_wav_path(self, options: TTSRequestOptions | None = None) -> str | None:
        request_options = options or TTSRequestOptions()
        if not request_options.reference_wav_path:
            return None
        candidate = Path(request_options.reference_wav_path).expanduser()
        if not candidate.exists():
            return None
        return str(candidate)

    def _model_is_v2(self) -> bool:
        tts_model = getattr(self._model, "tts_model", None)
        return type(tts_model).__name__ == "VoxCPM2Model"

    def _resolve_ultimate_clone(
        self,
        final_prompt_wav: str | None,
        final_prompt_text: str | None,
        final_reference_wav: str | None,
    ) -> str | None:
        """Ultimate-clone convergence for VoxCPM2.

        When running a VoxCPM2 model with a complete prompt pair
        (audio + text) and no explicit ``reference_wav_path``, reuse the
        prompt audio as the reference so generation enters the
        ``ref_continuation`` (ultimate cloning) path. VoxCPM1.5 has no
        reference channel, so it is left untouched.
        """
        if final_reference_wav is not None:
            return final_reference_wav
        if not (final_prompt_wav and final_prompt_text):
            return None
        if not self._model_is_v2():
            return None
        return final_prompt_wav

    def warmup(self, options: TTSRequestOptions | None = None, force: bool = False) -> dict[str, float | bool | int | None]:
        request_options = options or TTSRequestOptions()
        resolved_prompt_wav, resolved_prompt_text = self._resolve_reference_inputs(request_options)
        resolved_reference_wav = self._resolve_reference_wav_path(request_options)
        resolved_version = self._normalize_model_version(request_options.model_version)
        signature = (
            resolved_version,
            request_options.lora_selection,
            resolved_prompt_wav,
            resolved_prompt_text,
            resolved_reference_wav,
        )
        if self._warmup_result is not None and self._warmup_signature == signature and not force:
            return {
                **self._warmup_result,
                "cached": True,
            }

        total_started_at = perf_counter()
        model_loaded_before = self._model is not None
        lora_before = self._active_lora_path

        model_prepare_started_at = perf_counter()
        self._apply_lora(request_options.lora_selection, model_version=resolved_version)
        model = self.get_model(
            lora_ready=bool(request_options.lora_selection),
            model_version=resolved_version,
        )
        sample_rate = int(model.tts_model.sample_rate)
        tts_model_load_ms = (perf_counter() - model_prepare_started_at) * 1000.0
        if model_loaded_before and lora_before == self._active_lora_path:
            tts_model_load_ms = 0.0

        inference_started_at = perf_counter()
        generated_chunks = 0
        generated_samples = 0
        sanitized_options = TTSRequestOptions(
            model_version=resolved_version,
            lora_selection=request_options.lora_selection,
            prompt_audio_path=resolved_prompt_wav,
            prompt_text=resolved_prompt_text,
            reference_wav_path=resolved_reference_wav,
            cfg_value=request_options.cfg_value,
            inference_timesteps=request_options.inference_timesteps,
            seed=request_options.seed,
        )
        for chunk in self.stream_tts("你好。", options=sanitized_options):
            generated_chunks += 1
            generated_samples += int(getattr(chunk, "size", 0))
        tts_inference_ms = (perf_counter() - inference_started_at) * 1000.0

        result = {
            "cached": False,
            "tts_model_load_ms": round(tts_model_load_ms, 2),
            "tts_inference_ms": round(tts_inference_ms, 2),
            "tts_sample_rate": sample_rate,
            "tts_generated_chunks": generated_chunks,
            "tts_generated_samples": generated_samples,
            "total_ms": round((perf_counter() - total_started_at) * 1000.0, 2),
        }
        self._warmup_signature = signature
        self._warmup_result = result
        return result

    def stream_tts(
        self,
        text: str,
        prompt_wav_path: str | None = None,
        prompt_text: str | None = None,
        options: TTSRequestOptions | None = None,
    ) -> Generator[np.ndarray, None, None]:
        request_options = options or TTSRequestOptions()
        final_prompt_wav, final_prompt_text = self._resolve_reference_inputs(
            request_options,
            prompt_wav_path=prompt_wav_path,
            prompt_text=prompt_text,
        )
        final_reference_wav = self._resolve_reference_wav_path(request_options)
        cfg_value = request_options.cfg_value if request_options.cfg_value is not None else self.config.cfg_value
        inference_timesteps = (
            request_options.inference_timesteps
            if request_options.inference_timesteps is not None
            else self.config.inference_timesteps
        )

        self._apply_lora(request_options.lora_selection, model_version=request_options.model_version)
        model = self.get_model(
            lora_ready=bool(request_options.lora_selection),
            model_version=request_options.model_version,
        )
        final_reference_wav = self._resolve_ultimate_clone(
            final_prompt_wav, final_prompt_text, final_reference_wav
        )
        self._apply_seed(request_options.seed)
        yield from model.generate_streaming(
            text=text,
            prompt_wav_path=final_prompt_wav,
            prompt_text=final_prompt_text,
            reference_wav_path=final_reference_wav,
            cfg_value=float(cfg_value),
            inference_timesteps=int(inference_timesteps),
            min_len=self.config.min_len,
            max_len=self.config.max_len,
            normalize=self.config.normalize,
            denoise=self.config.denoise,
            seed=request_options.seed,
        )
