# Build-time weight bake for the SoulX-Singer SVC worker.
#
# ⚠️ Качаем НЕ репозитории целиком, а ровно те веса, которые загружает handler.
#
# Замер 18.08.2026 (HF API, размеры файлов):
#   SoulX-Singer            5.61 ГБ = model.pt 2.82 (TTS, НЕ используется)
#                                   + model-svc.pt 2.79 (используется)
#   SoulX-Singer-Preprocess 6.92 ГБ = parakeet 2.47 + караоке-сепаратор 1.72
#                                   + paraformer 0.99 + дереверб 0.91
#                                   + rosvot 0.60 + rmvpe 0.18 (используется)
#
# То есть из 12.5 ГБ нужно 3.0. Всё остальное — инструменты расшифровки текста и
# MIDI из препроцесс-пайплайна апстрима, которых запрос на конверсию не касается
# (это же сказано в requirements-soulx.txt про выброшенный nemo).
#
# Цена лишнего оказалась не абстрактной: образ вырос до 16.3 ГБ сжатого, перестал
# влезать в дисковую квоту воркера (25 ГБ) — и SoulX трое суток не мог
# запуститься ВООБЩЕ. Симптом при этом был неотличим от поломки GPU: воркеры
# вечно «initializing», задачи в очереди, счёт капает. См. разбор 18.08.
#
# Handler грузит ровно два файла:
#   pretrained_models/SoulX-Singer/model-svc.pt
#   pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt
# Мелкие файлы (конфиги, словари) оставляем целиком — они ничего не весят, а
# отрезать их значит гадать, что ещё читает load_config().
#
# NB: snapshot_download напрямую, а не через `hf` CLI. CLI живёт в
# huggingface_hub >= 1.0, а его установка ломает transformers 4.41.2, которому
# нужен hub < 1.0. Проверено на спайке 2026-07-08.
from huggingface_hub import snapshot_download

snapshot_download(
    "Soul-AILab/SoulX-Singer",
    local_dir="/soulx/pretrained_models/SoulX-Singer",
    # Всё, КРОМЕ TTS-модели: она весит столько же, сколько нужная нам SVC.
    ignore_patterns=["model.pt"],
)

snapshot_download(
    "Soul-AILab/SoulX-Singer-Preprocess",
    local_dir="/soulx/pretrained_models/SoulX-Singer-Preprocess",
    # Из препроцесса нужен только извлекатель F0. Остальное — ASR, разделение
    # вокала, дереверберация и MIDI: 6.7 ГБ, которых конверсия не касается.
    ignore_patterns=[
        "parakeet-tdt-0.6b-v2/*",
        "mel-band-roformer-karaoke/*",
        "dereverb_mel_band_roformer/*",
        "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/*",
        "rosvot/*",
    ],
)
print("soulx weights baked (svc + rmvpe only)", flush=True)
