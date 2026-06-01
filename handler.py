"""
RunPod Serverless handler для YingMusic-SVC — zero-shot пение голосом юзера.
Берёт вокал (source) + образец голоса юзера (reference) → вокал тембром юзера.

Вход (event["input"]):
  source_audio     — base64 исходного вокала/песни (напр. результат ACE-Step)
  reference_audio  — base64 образца голоса юзера (timbre reference)
  accompany_audio  — base64 аккомпанемента (опционально, для ремикса)
  diffusion_steps  — по умолч. 100 (выше = качественнее/медленнее)

Выход: audio_base64 (wav) либо error + stdout/stderr/traceback.

Пути checkpoint/config можно переопределить env: YMSVC_CHECKPOINT, YMSVC_CONFIG.
"""
import os
import glob
import base64
import tempfile
import subprocess
import traceback

import runpod

REPO = "/app/YingMusic-SVC"
CONFIG = os.environ.get("YMSVC_CONFIG", os.path.join(REPO, "configs", "YingMusic-SVC.yml"))


def _find_ckpt():
    env = os.environ.get("YMSVC_CHECKPOINT")
    if env and os.path.exists(env):
        return env
    cands = glob.glob(os.path.join(REPO, "pretrained", "**", "*.pth"), recursive=True)
    # отдаём предпочтение файлу с 'full'/'svc' в имени
    cands.sort(key=lambda p: (("full" not in p.lower()), ("svc" not in p.lower()), p))
    return cands[0] if cands else None


CKPT = _find_ckpt()


def _w(path, b64):
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64))


def handler(event):
    try:
        inp = event.get("input", {}) or {}
        src_b64 = inp.get("source_audio")
        ref_b64 = inp.get("reference_audio")
        separate_vocals = bool(inp.get("separate_vocals", False))  # отделять вокал от музыки (TODO-стадия)
        if not src_b64 or not ref_b64:
            return {"error": "need source_audio and reference_audio (base64)"}
        if not CKPT:
            return {"error": "checkpoint .pth не найден в /app/YingMusic-SVC/pretrained",
                    "config": CONFIG}

        steps = int(inp.get("diffusion_steps", 100))
        wd = tempfile.mkdtemp()
        s = os.path.join(wd, "source.wav")
        t = os.path.join(wd, "target.wav")
        _w(s, src_b64)
        _w(t, ref_b64)

        cmd = [
            "python", "my_inference.py",
            "--source", s,
            "--target", t,
            "--diffusion-steps", str(steps),
            "--checkpoint", CKPT,
            "--config", CONFIG,
            "--expname", "job",
            "--cuda", "0",
            "--fp16", "True",
        ]
        if inp.get("accompany_audio"):
            a = os.path.join(wd, "accompany.wav")
            _w(a, inp["accompany_audio"])
            cmd += ["--accompany", a]

        print("[YingMusic] ckpt:", CKPT, flush=True)
        print("[YingMusic] run:", " ".join(cmd), flush=True)
        p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=1200)

        if p.returncode != 0:
            return {"error": "yingmusic failed", "stdout": p.stdout[-3000:], "stderr": p.stderr[-3000:]}

        # результат в ./outputs/job/*.wav (относительно REPO)
        out_glob = os.path.join(REPO, "outputs", "job", "**", "*.wav")
        files = sorted(glob.glob(out_glob, recursive=True), key=os.path.getmtime)
        if not files:
            return {"error": "no output file", "stdout": p.stdout[-3000:], "stderr": p.stderr[-3000:]}

        with open(files[-1], "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        # converted_audio — согласованное имя для A/B с HQ-SVC; audio_base64 — для совместимости
        return {
            "converted_audio": b64,
            "audio_base64": b64,
            "format": "wav",
            "model": "yingmusic-svc",
            "separate_vocals": separate_vocals,
            "logs": p.stdout[-1500:],
        }

    except Exception:
        return {"error": "handler crashed", "traceback": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
