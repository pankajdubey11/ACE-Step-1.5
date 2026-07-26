"""RunPod Serverless handler — ACE-Step 1.5 Background Music Engine (Phase 3).

Zero external storage: returns the generated audio as a Base64 string.

Operation: "background_music"
  input: {
    "music_prompt": "Cinematic dramatic orchestral, slow tempo, suspense",  # required
    "duration_seconds": 30,          # target length
    "lyrics": "",                    # "" -> instrumental BGM (default)
    "seed": -1,
    "language": "en",
    "inference_steps": 8,
    "guidance_scale": 1.0,
    "audio_format": "mp3"            # mp3 | wav | flac | opus | aac
  }
  output: { "audio_base64": "<mp3 b64>", "meta": {...} }

Both model handlers (DiT + LLM) are warm-loaded into VRAM at container startup.
On CUDA use LLM backend "vllm" (fast) or "hf" (portable) — never "mlx" (Apple).
"""

from __future__ import annotations

import base64
import os
import sys
import tempfile
import time
import traceback

os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)

import runpod

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

# ----------------------------- configuration --------------------------------
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", os.path.join(_PROJECT_ROOT, "checkpoints"))
DIT_CONFIG = os.environ.get("DIT_CONFIG", "acestep-v15-turbo")
LM_MODEL = os.environ.get("LM_MODEL", "acestep-5Hz-lm-0.6B")
LM_BACKEND = os.environ.get("LM_BACKEND", "hf")   # 'vllm' (fast, CUDA) | 'hf' (portable)
OFFLOAD = os.environ.get("OFFLOAD_TO_CPU", "0") == "1"
DEFAULT_STEPS = int(os.environ.get("INFERENCE_STEPS", "8"))

_DIT = None
_LLM = None


def get_handlers():
    """Warm-load DiT + LLM handlers once (VRAM-resident)."""
    global _DIT, _LLM
    if _DIT is None:
        print(f"[warm-start] init DiT handler ({DIT_CONFIG})...", flush=True)
        dit = AceStepHandler()
        msg, ok = dit.initialize_service(
            project_root=_PROJECT_ROOT, config_path=DIT_CONFIG,
            device="auto", offload_to_cpu=OFFLOAD,
        )
        if not ok:
            raise RuntimeError(f"DiT init failed: {msg}")
        _DIT = dit
        print(f"[warm-start] DiT ready — {msg}", flush=True)
    if _LLM is None:
        print(f"[warm-start] init LLM handler ({LM_MODEL}, backend={LM_BACKEND})...", flush=True)
        llm = LLMHandler()
        msg, ok = llm.initialize(
            checkpoint_dir=CHECKPOINT_DIR, lm_model_path=LM_MODEL,
            backend=LM_BACKEND, device="auto", offload_to_cpu=OFFLOAD, dtype=None,
        )
        if not ok:
            raise RuntimeError(f"LLM init failed: {msg}")
        _LLM = llm
        print(f"[warm-start] LLM ready — {msg}", flush=True)
    return _DIT, _LLM


def _read_result_audio(result, audio_format: str) -> str:
    """Return the first generated audio as Base64 (from path or in-memory bytes)."""
    if not getattr(result, "success", False):
        raise RuntimeError(getattr(result, "status_message", "generation failed"))
    audios = getattr(result, "audios", None) or []
    if not audios:
        raise RuntimeError("no audio in result")
    a = audios[0]
    # Prefer a written file path; fall back to in-memory bytes if present.
    path = a.get("path") if isinstance(a, dict) else None
    if path and os.path.exists(path):
        with open(path, "rb") as fh:
            return base64.b64encode(fh.read()).decode("utf-8")
    for key in ("bytes", "audio_bytes", "data"):
        if isinstance(a, dict) and a.get(key):
            raw = a[key]
            if isinstance(raw, str):
                return raw  # already base64
            return base64.b64encode(raw).decode("utf-8")
    raise RuntimeError("audio produced but no path/bytes to read")


def op_background_music(inp: dict) -> dict:
    prompt = inp.get("music_prompt") or inp.get("prompt")
    if not prompt:
        raise ValueError("background_music requires 'music_prompt'")
    duration = inp.get("duration_seconds") or inp.get("duration")
    audio_format = (inp.get("audio_format") or "mp3").lower()
    dit, llm = get_handlers()

    with tempfile.TemporaryDirectory() as save_dir:
        params = GenerationParams(
            task_type="text2music",
            thinking=bool(inp.get("thinking", True)),
            caption=prompt,
            lyrics=inp.get("lyrics", ""),           # "" -> instrumental BGM
            bpm=inp.get("bpm"),
            keyscale=inp.get("keyscale", ""),
            timesignature=inp.get("timesignature", ""),
            vocal_language=inp.get("language", "en"),
            duration=duration,
            inference_steps=int(inp.get("inference_steps", DEFAULT_STEPS)),
            guidance_scale=float(inp.get("guidance_scale", 1.0)),
            seed=int(inp.get("seed", -1)),
        )
        config = GenerationConfig(
            batch_size=1,
            audio_format=audio_format,
            mp3_bitrate=inp.get("mp3_bitrate", "192k"),
        )
        result = generate_music(dit, llm, params=params, config=config, save_dir=save_dir)
        audio_b64 = _read_result_audio(result, audio_format)

    return {"audio_base64": audio_b64,
            "meta": {"operation": "background_music", "format": audio_format,
                     "duration_seconds": duration, "instrumental": not inp.get("lyrics"),
                     "prompt": prompt[:120]}}


_OPS = {"background_music": op_background_music}


def handler(job: dict) -> dict:
    inp = job.get("input") or {}
    op = (inp.get("operation") or "background_music").strip()
    if op not in _OPS:
        return {"error": f"unknown operation {op!r}. expected {sorted(_OPS)}"}
    started = time.time()
    try:
        result = _OPS[op](inp)
        result.setdefault("meta", {})["compute_seconds"] = round(time.time() - started, 2)
        return result
    except Exception as exc:
        return {"error": str(exc), "trace": traceback.format_exc()[-1500:], "operation": op}


# Warm-load both handlers into VRAM at container startup (not per request).
if os.environ.get("PRELOAD", "1") == "1":
    try:
        get_handlers()
    except Exception as exc:
        print(f"[warm-start] deferred (will load on first call): {exc}", flush=True)

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
