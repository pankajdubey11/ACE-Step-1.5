# TrimTime Music Engine — ACE-Step 1.5 on RunPod Serverless (Phase 3)

Self-hosted royalty-free cinematic BGM. **Zero external storage** — returns the
audio as Base64.

## Endpoint: `background_music`
`POST https://api.runpod.io/v2/<MUSIC_ENDPOINT_ID>/runsync` — `Authorization: Bearer <RUNPOD_API_KEY>`

```json
{ "input": {
  "operation": "background_music",
  "music_prompt": "Cinematic dramatic orchestral, slow tempo, suspense",
  "duration_seconds": 30,
  "lyrics": "",                 // "" -> instrumental BGM (default)
  "audio_format": "mp3",        // mp3 | wav | flac | opus | aac
  "seed": -1, "language": "en", "inference_steps": 8, "guidance_scale": 1.0
}}
```

## Response
```json
{ "audio_base64": "<mp3 b64>",
  "meta": { "format": "mp3", "duration_seconds": 30, "instrumental": true,
            "compute_seconds": 9.4 } }
```
Errors: `{ "error": "...", "trace": "...", "operation": "background_music" }`.

## Deploy config (env)
| Env | Default | Notes |
|-----|---------|-------|
| `DIT_CONFIG` | `acestep-v15-turbo` | DiT model config |
| `LM_MODEL` | `acestep-5Hz-lm-0.6B` | reasoning LM |
| `LM_BACKEND` | `hf` | **CUDA: `vllm` (fast) or `hf` (portable) — never `mlx`** |
| `INFERENCE_STEPS` | `8` | turbo default |
| `CHECKPOINT_DIR` | `/app/checkpoints` | point to a network volume to persist weights |
| `PRELOAD` | `1` | warm-load DiT+LLM into VRAM at startup |

**CUDA 12.8 / Python 3.11** (repo requirement). Original repo Dockerfile kept as
`Dockerfile.repo-original`.
