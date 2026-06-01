"""
RunPod Serverless handler для YingMusic-SVC — zero-shot пение голосом юзера.

Пайплайн (для чистого результата):
  source (полная песня) --[bs_roformer]--> чистый вокал + аккомпанемент
  чистый вокал + reference (голос юзера) --[YingMusic]--> вокал тембром юзера
  (+ ремикс с аккомпанементом через --accompany)

Вход (event["input"]):
  source_audio     — base64 исходной песни/вокала (напр. ACE-Step)
  reference_audio  — base64 образца голоса юзера
  separate_vocals  — отделять вокал от музыки (по умолч. True; для уже чистого вокала ставь False)
  diffusion_steps  — по умолч. 100

Выход: converted_audio (wav) + logs, либо error + stdout/stderr.
"""
import os
import glob
import base64
import shutil
import tempfile
import subprocess
import traceback

import runpod

REPO = "/app/YingMusic-SVC"
SEP_DIR = os.path.join(REPO, "accom_separation")
CONFIG = os.environ.get("YMSVC_CONFIG", os.path.join(REPO, "configs", "YingMusic-SVC.yml"))


def _find_main_ckpt():
    env = os.environ.get("YMSVC_CHECKPOINT")
    if env and os.path.exists(env):
        return env
    preferred = os.path.join(REPO, "pretrained", "YingMusic-SVC-full.pt")
    if os.path.exists(preferred):
        return preferred
    cands = []
    for ext in ("*.pt", "*.pth", "*.ckpt", "*.safetensors"):
        cands += glob.glob(os.path.join(REPO, "pretrained", "**", ext), recursive=True)
    cands = [c for c in cands if "roformer" not in os.path.basename(c).lower()]
    cands.sort(key=lambda p: (("full" not in p.lower()), ("svc" not in p.lower()), p))
    return cands[0] if cands else None


def _find_roformer_ckpt():
    cands = glob.glob(os.path.join(REPO, "pretrained", "**", "*roformer*.ckpt"), recursive=True)
    return cands[0] if cands else None


def _find_roformer_config():
    cands = glob.glob(os.path.join(SEP_DIR, "**", "*roformer*.yaml"), recursive=True) \
        + glob.glob(os.path.join(SEP_DIR, "**", "*roformer*.yml"), recursive=True)
    return cands[0] if cands else None


CKPT = _find_main_ckpt()


def _w(path, b64):
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64))


def _separate_vocals(src_path, workdir):
    """Через bs_roformer выделяем вокал и аккомпанемент. Возвращает (vocal, accompaniment) или (None, None)."""
    rof_ckpt = _find_roformer_ckpt()
    rof_cfg = _find_roformer_config()
    if not rof_ckpt or not rof_cfg or not os.path.exists(os.path.join(SEP_DIR, "inference.py")):
        return None, None, "separation skipped (no roformer ckpt/config/inference.py)"

    in_dir = os.path.join(workdir, "sep_in")
    out_dir = os.path.join(workdir, "sep_out")
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(src_path, os.path.join(in_dir, "song.wav"))

    cmd = [
        "python", "inference.py",
        "--model_type", "bs_roformer",
        "--config_path", rof_cfg,
        "--start_check_point", rof_ckpt,
        "--input_folder", in_dir,
        "--store_dir", out_dir,
        "--extract_other",
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONWARNINGS="ignore")
    p = subprocess.run(cmd, cwd=SEP_DIR, capture_output=True, text=True, timeout=600, env=env)
    if p.returncode != 0:
        return None, None, f"separation failed: {p.stderr[-1500:]}"

    wavs = glob.glob(os.path.join(out_dir, "**", "*.wav"), recursive=True)
    vocal = next((w for w in wavs if "vocal" in os.path.basename(w).lower()), None)
    accomp = next((w for w in wavs if any(k in os.path.basename(w).lower()
                                          for k in ("other", "instrum", "accomp", "no_vocal"))), None)
    return vocal, accomp, p.stdout[-800:]


def handler(event):
    try:
        if not CKPT:
            listing = glob.glob(os.path.join(REPO, "pretrained", "**", "*"), recursive=True)
            return {"error": "main checkpoint не найден", "config": CONFIG, "files": listing[:50]}

        inp = event.get("input", {}) or {}
        src_b64 = inp.get("source_audio")
        ref_b64 = inp.get("reference_audio")
        if not src_b64 or not ref_b64:
            return {"error": "need source_audio and reference_audio (base64)"}
        separate = bool(inp.get("separate_vocals", True))
        steps = int(inp.get("diffusion_steps", 100))

        wd = tempfile.mkdtemp()
        src = os.path.join(wd, "source.wav")
        ref = os.path.join(wd, "target.wav")
        _w(src, src_b64)
        _w(ref, ref_b64)

        sep_log = ""
        vocal_src = src
        accompany = None
        if separate:
            v, a, sep_log = _separate_vocals(src, wd)
            if v:
                vocal_src = v
                accompany = a

        cmd = [
            "python", "my_inference.py",
            "--source", vocal_src,
            "--target", ref,
            "--diffusion-steps", str(steps),
            "--checkpoint", CKPT,
            "--config", CONFIG,
            "--expname", "job",
            "--cuda", "0",
            "--fp16", "True",
        ]
        if accompany:
            cmd += ["--accompany", accompany]

        print("[YingMusic] separate:", separate, "| accompany:", bool(accompany), flush=True)
        print("[YingMusic] run:", " ".join(cmd), flush=True)
        p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=1200)
        if p.returncode != 0:
            return {"error": "yingmusic failed", "sep_log": sep_log,
                    "stdout": p.stdout[-3000:], "stderr": p.stderr[-3000:]}

        files = sorted(glob.glob(os.path.join(REPO, "outputs", "job", "**", "*.wav"), recursive=True),
                       key=os.path.getmtime)
        if not files:
            return {"error": "no output file", "sep_log": sep_log,
                    "stdout": p.stdout[-3000:], "stderr": p.stderr[-3000:]}

        with open(files[-1], "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return {
            "converted_audio": b64,
            "audio_base64": b64,
            "format": "wav",
            "model": "yingmusic-svc",
            "separated": bool(accompany),
            "logs": p.stdout[-1200:],
        }

    except Exception:
        return {"error": "handler crashed", "traceback": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
