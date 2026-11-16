# Octave repair for the F0 curve seed-vc's singing model is conditioned on.
#
# WHY. seed-vc reads the melody with RMVPE, and on singing voices whose even harmonics are
# stronger than the odd ones RMVPE reads an octave too HIGH: measured 2026-11-16 on 9 songs,
# 1.5-12.7 % of voiced frames (male voices worst). The model then sings that wrong octave —
# the output's fundamental sits 36 dB under its second harmonic there: an audible jump up
# («петухи»). The physics settles who is right: a voice at f has energy at f, 3f, 5f; a voice
# at 2f has none there. In 90 % of the disputed frames the odd harmonics of the lower
# candidate stand ~16 dB above the between-harmonic floor.
#
# RULE. Keep RMVPE. Where a second, fast tracker reads an octave apart, take the reading the
# harmonic series confirms. Frames where the trackers agree are never touched.
import numpy as np
import librosa

SR = 16000
HOP = 160          # RMVPE's grid: 10 ms at 16 kHz
N_FFT = 4096       # 3.9 Hz bins: resolves the odd harmonics of a 60 Hz voice
MARGIN_DB = 6.0


def _level(col, freqs, f):
    w = max(f * (2 ** (0.4 / 12) - 1), (freqs[1] - freqs[0]) * 1.01)
    m = (freqs > f - w) & (freqs < f + w)
    return 20 * np.log10(col[m].max() + 1e-12) if m.any() else -240.0


def lower_is_real(col, freqs, g, margin=MARGIN_DB):
    """Is g (and not 2g) the pitch? Odd harmonics of g above the bins between harmonics."""
    if 5 * g >= SR / 2:
        return False
    odd = np.mean([_level(col, freqs, k * g) for k in (1, 3, 5)])
    gap = np.mean([_level(col, freqs, k * g) for k in (1.5, 2.5, 3.5)])
    return odd - gap > margin


def second_opinion(y, kind):
    if kind == 'pyin':
        f, v, _ = librosa.pyin(y, fmin=50, fmax=1100, sr=SR, frame_length=1024, hop_length=HOP,
                               center=True, resolution=0.5)
        return np.where(v & np.isfinite(f), f, 0.0)
    f = librosa.yin(y, fmin=50, fmax=1100, sr=SR, frame_length=2048, hop_length=HOP, center=True)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP, center=True)[0]
    return np.where(rms[:len(f)] > 1e-4, f, 0.0)


def fix_octaves(y, f0, kind='yin'):
    """y: mono float32 at 16 kHz; f0: RMVPE curve (Hz, 0 = unvoiced) on the 10 ms grid."""
    y = np.asarray(y, dtype=np.float32)
    f0 = np.asarray(f0)
    out = f0.astype(np.float64).copy()
    other = second_opinion(y, kind)
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP, center=True))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    n = min(len(out), len(other), S.shape[1])
    changed = 0
    for i in range(n):
        r, o = out[i], other[i]
        if r <= 1 or o <= 1:
            continue
        st = 12 * np.log2(r / o)
        if 10 < st < 14 and lower_is_real(S[:, i], freqs, o):
            out[i] = o; changed += 1
        elif -14 < st < -10 and not lower_is_real(S[:, i], freqs, r):
            out[i] = o; changed += 1
    fix_octaves.last_changed = changed
    return out.astype(f0.dtype)


fix_octaves.last_changed = 0
