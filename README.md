# YingMusic-SVC — Serverless-воркер (пение голосом юзера, SOTA дек 2025)

Zero-shot перекраска вокала в тембр юзера, заточена под реальные песни
(подавляет аккомпанемент/хор/реверб, чистые высокие ноты). Без обучения под юзера.

## Деплой (как ACE-Step — через GitHub)
1. Новый репозиторий GitHub, напр. `yingmusic-runpod-worker`.
2. Залей в корень: `handler.py`, `Dockerfile`, `README.md`.
3. RunPod → Serverless → New Endpoint → Deploy from GitHub → ветка `main`, Dockerfile `Dockerfile`.
4. Настройки:
   - **GPU: RTX 4090** (если будет мало — поднять до 48 ГБ).
   - **Min Workers: 0**, Max 2–3, Idle 60–120с, FlashBoot ON, Disk 30 ГБ.
   - **Execution timeout: 600+ сек** (2-стадийный пайплайн).
5. Endpoint ID → в `.env` как `RUNPOD_ENDPOINT_RVC`.

⚠️ Сборка: качает веса с HuggingFace (`GiantAILab/YingMusic-SVC`) — может идти 15–30 мин.
В Build logs ищи `CKPT DOWNLOADED` и `RUNPOD OK`.

## Тестовый запрос
```json
{
  "input": {
    "source_audio": "<base64 вокала-источника (напр. ACE-Step)>",
    "reference_audio": "<base64 образца голоса юзера>",
    "diffusion_steps": 100
  }
}
```
Ответ — `audio_base64` (wav). При сбое handler вернёт `stdout`/`stderr` — присылай их.

## Ожидаемые места отладки (research-код)
- Имя/путь чекпойнта (`.pth`) — handler ищет в `pretrained/`, можно задать env `YMSVC_CHECKPOINT`.
- Путь config — по умолч. `configs/YingMusic-SVC.yml` (env `YMSVC_CONFIG`).
- Папка вывода — ожидаем `outputs/job/`.
- Возможны недостающие под-модели (rmvpe, campplus, BigVGAN, Whisper) — докачиваются в рантайме.

## Место в пайплайне
ACE-Step (песня) → **YingMusic-SVC** (голос юзера) → видео + липсинк.
