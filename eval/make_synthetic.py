"""Make a SYNTHETIC test set for phrase matching when no real dysarthric recordings are at hand.

Windows' built-in voices say each phrase several times at different speeds; each copy is then
distorted the way dysarthric speech and real rooms distort it: uneven speed within the phrase,
muffled consonants (low-pass), breathy noise and background noise (fan hum or babble).

This is a smoke test for the matcher, not evidence about patients: real dysarthria changes
articulation in ways these effects do not imitate. Use UA-Speech or TORGO for real numbers.

Usage:  python -m eval.make_synthetic [--out eval/data/synthetic] [--repeats 5]   (Windows only)
Writes: <out>/<voice>/<phrase>/<n>.wav
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from backend import dsp

PHRASES = ["water", "food", "pain", "toilet", "yes", "no", "help", "doctor", "medicine", "hot", "cold",
           "tired", "sleep", "thank you", "turn me over", "sit up", "light off", "call my son",
           "paani", "dawai", "haan", "nahi", "bathroom", "blanket"]
RATES = [-3, -1, 0, 1, -2, 2, -4]


def synthesize(out: Path, repeats: int) -> list[Path]:
    """One PowerShell call renders every phrase/voice/rate with System.Speech."""
    raw = out / "_raw"
    raw.mkdir(parents=True, exist_ok=True)
    script = f"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$phrases = @({",".join(f"'{p}'" for p in PHRASES)})
$rates = @({",".join(str(r) for r in RATES[:repeats])})
foreach ($v in $s.GetInstalledVoices()) {{
  $name = $v.VoiceInfo.Name -replace '^Microsoft ', '' -replace ' Desktop$', ''
  $s.SelectVoice($v.VoiceInfo.Name)
  foreach ($p in $phrases) {{
    for ($i = 0; $i -lt $rates.Count; $i++) {{
      $s.Rate = $rates[$i]
      $f = Join-Path '{raw}' ($name + '__' + ($p -replace ' ', '_') + '__' + $i + '.wav')
      $s.SetOutputToWaveFile($f, $fmt); $s.Speak($p); $s.SetOutputToNull()
    }}
  }}
}}
"""
    with tempfile.NamedTemporaryFile("w", suffix=".ps1", delete=False, encoding="utf-8") as f:
        f.write(script)
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", f.name], check=True)
    Path(f.name).unlink()
    return sorted(raw.glob("*.wav"))


def uneven_speed(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Slow down by a different amount in each ~150 ms piece (0.55x-1.0x speed)."""
    seg = 2400
    parts = []
    for i in range(0, len(x), seg):
        piece = x[i:i + seg]
        speed = rng.uniform(0.55, 1.0)
        n = max(2, int(len(piece) / speed))
        parts.append(np.interp(np.linspace(0, len(piece) - 1, n), np.arange(len(piece)), piece))
    return np.concatenate(parts).astype(np.float32)


def lowpass(x: np.ndarray, cutoff: float) -> np.ndarray:
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / dsp.SR)
    spec *= 1 / (1 + (f / cutoff) ** 6)
    return np.fft.irfft(spec, n=len(x)).astype(np.float32)


def background(n: int, kind: str, rng: np.random.Generator) -> np.ndarray:
    w = rng.standard_normal(n)
    if kind == "fan":   # low rumble + hum
        b = np.cumsum(w)
        b -= np.convolve(b, np.ones(400) / 400, mode="same")
        b += 0.5 * np.sin(2 * np.pi * 120 * np.arange(n) / dsp.SR) * np.std(b)
        return b
    # babble-like: noise shaped by a slowly varying speech-band envelope
    env = np.abs(np.convolve(rng.standard_normal(n), np.ones(1600) / 1600, mode="same"))
    return lowpass(w.astype(np.float32), 3000) * (0.3 + env / (env.max() + 1e-9))


def distort(x: np.ndarray, rng: np.random.Generator, severity: float) -> np.ndarray:
    x = dsp.trim_silence(x)
    x = uneven_speed(x, rng) if severity > 0 else x
    x = lowpass(x, rng.uniform(2200, 3800)) if severity > 0.3 else x
    x = x + severity * 0.15 * np.abs(x).mean() * rng.standard_normal(len(x)) * (np.abs(x) > 0.01)   # breathiness
    pad = np.zeros(int(0.3 * dsp.SR), np.float32)
    x = np.concatenate([pad, x, pad])
    snr_db = rng.uniform(5, 20)
    noise = background(len(x), rng.choice(["fan", "babble"]), rng)
    noise *= np.sqrt(np.mean(x ** 2) / (np.mean(noise ** 2) + 1e-12) / 10 ** (snr_db / 10))
    y = x + noise
    return (0.5 * y / (np.abs(y).max() + 1e-9)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "data" / "synthetic")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    files = synthesize(a.out, a.repeats)
    for f in files:
        voice, phrase, i = f.stem.split("__")
        x, sr = dsp.read_wav(f.read_bytes())
        # the first two copies act as enrolled templates (milder), the rest as harder live turns
        y = distort(dsp.resample(x, sr), rng, severity=0.4 if int(i) < 2 else 1.0)
        dest = a.out / voice / phrase / f"{i}.wav"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(dsp.write_wav(y))
        f.unlink()
    (a.out / "_raw").rmdir()
    print(f"wrote {len(files)} clips to {a.out}")


if __name__ == "__main__":
    main()
