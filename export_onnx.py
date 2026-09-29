"""
Export the trained PyTorch models (.pt) to ONNX so the web app can run them
inside the browser (free static hosting, no Python server needed).

Usage (from the project folder, where models/ and results/ are):
    pip install onnx onnxscript onnxruntime
    python export_onnx.py

It writes:
    web/models/<id>.onnx      one file per model
    web/models/manifest.json  classes, normalization (mu/sd), UAR, mel filterbank

Models whose .pt file does not exist yet are skipped, so the English (RAVDESS)
entries below can stay: when those files appear, run the script again.
"""
import json
import sys
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# Models to export. Add or remove lines here.
#   id      unique name (file name in web/models/)
#   pt      checkpoint saved by the notebook: {"model", "mu", "sd", "classes"}
#   corpus  training corpus, shown in the app
#   lang    language code of the corpus: de, en, es, pt, fr, multi
#   metrics metrics.json written by the cross-validation cell (optional)
#   key     entry of that model inside metrics.json
# ----------------------------------------------------------------------------
MODELS = [
    dict(id="emodb_scratch", pt="models/scratch.pt", corpus="EmoDB", lang="de",
         metrics="results/metrics.json", key="scratch"),
    dict(id="emodb_resnet18", pt="models/resnet18.pt", corpus="EmoDB", lang="de",
         metrics="results/metrics.json", key="resnet18"),
    dict(id="ravdess_scratch", pt="models/ravdess/scratch.pt", corpus="RAVDESS", lang="en",
         metrics="results/ravdess/metrics.json", key="scratch"),
    dict(id="ravdess_resnet18", pt="models/ravdess/resnet18.pt", corpus="RAVDESS", lang="en",
         metrics="results/ravdess/metrics.json", key="resnet18"),
]
OUT = Path("web/models")

# Feature parameters: IDENTICAL to the notebook
SR, N_SAMPLES, WIN, HOP, N_FFT, N_MELS = 16000, 3 * 16000, 400, 160, 512, 64
TRIM_DB = 30


# ----------------------------------------------------------------------------
# NumPy copy of what web/dsp.js does. Used to prove that the browser computes
# the same log-mel as librosa (check_dsp below).
# ----------------------------------------------------------------------------
def ref_trim(y, top_db=TRIM_DB, frame_length=2048, hop=512):
    ypad = np.pad(y.astype(np.float64), frame_length // 2)
    n_frames = 1 + len(y) // hop
    ms = np.array([np.mean(ypad[f * hop:f * hop + frame_length] ** 2) for f in range(n_frames)])
    db = 10 * np.log10(np.maximum(1e-10, ms)) - 10 * np.log10(max(1e-10, ms.max()))
    nz = np.flatnonzero(db > -top_db)
    if nz.size == 0:
        return y[:0]
    return y[nz[0] * hop:min(len(y), (nz[-1] + 1) * hop)]


def ref_fix_length(x, n=N_SAMPLES):
    if len(x) < n:
        return np.pad(x, (0, n - len(x)))
    s = (len(x) - n) // 2
    return x[s:s + n]


def ref_logmel(x, fb):
    x = x.astype(np.float64)
    x = x / (np.abs(x).max() + 1e-9)
    x = ref_fix_length(x)
    ypad = np.pad(x, N_FFT // 2)
    win = np.zeros(N_FFT)
    off = (N_FFT - WIN) // 2
    win[off:off + WIN] = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WIN) / WIN)   # periodic Hann
    n_frames = 1 + (len(ypad) - N_FFT) // HOP
    frames = np.stack([ypad[t * HOP:t * HOP + N_FFT] for t in range(n_frames)], axis=1)
    power = np.abs(np.fft.rfft(frames * win[:, None], axis=0)) ** 2           # (257, 301)
    db = 10 * np.log10(np.maximum(1e-10, fb @ power))
    return np.maximum(db, db.max() - 80.0)                                       # (64, 301)


def check_dsp(fb):
    """Compare the browser algorithm with librosa on a synthetic voice-like signal."""
    import librosa
    rng = np.random.default_rng(0)
    t = np.arange(int(4.2 * SR)) / SR
    f0 = 140 + 60 * np.sin(2 * np.pi * 0.7 * t)
    voice = sum(np.sin(2 * np.pi * k * np.cumsum(f0) / SR) / k for k in range(1, 12))
    voice *= (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)) ** 2
    y = np.concatenate([np.zeros(SR // 2), voice, np.zeros(SR // 3)])
    y = (0.3 * y + 0.002 * rng.standard_normal(len(y))).astype(np.float32)

    lib_trim, _ = librosa.effects.trim(y, top_db=TRIM_DB)
    my_trim = ref_trim(y)
    assert abs(len(lib_trim) - len(my_trim)) <= 512, f"trim differs: {len(lib_trim)} vs {len(my_trim)}"

    x = lib_trim.astype(np.float32)
    x = x / (np.abs(x).max() + 1e-9)
    S = librosa.feature.melspectrogram(y=ref_fix_length(x), sr=SR, n_fft=N_FFT, win_length=WIN,
                                       hop_length=HOP, n_mels=N_MELS)
    lib = librosa.power_to_db(S, ref=1.0)
    mine = ref_logmel(lib_trim, fb)
    err = np.abs(lib - mine).max()
    print(f"  log-mel browser vs librosa: max error {err:.4f} dB")
    assert err < 0.05, "the browser preprocessing does not match librosa"


# ----------------------------------------------------------------------------
# Model definitions: same as the notebook
# ----------------------------------------------------------------------------
def build_model(state, n_classes):
    import torch.nn as nn
    import torchvision

    def conv_block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

    class ScratchCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(conv_block(1, 32), conv_block(32, 64),
                                          conv_block(64, 128), conv_block(128, 256))
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                      nn.Dropout(0.3), nn.Linear(256, n_classes))

        def forward(self, x):
            return self.head(self.features(x))

    class TransferResNet(nn.Module):
        def __init__(self, head_dropout):
            super().__init__()
            self.net = torchvision.models.resnet18(weights=None)
            fc = nn.Linear(self.net.fc.in_features, n_classes)
            self.net.fc = nn.Sequential(nn.Dropout(0.3), fc) if head_dropout else fc

        def forward(self, x):
            return self.net(x.repeat(1, 3, 1, 1))

    if any(k.startswith("features.") for k in state):
        model, arch = ScratchCNN(), "scratch"
    else:
        model, arch = TransferResNet("net.fc.1.weight" in state), "resnet18"
    model.load_state_dict(state)
    return model.eval(), arch


def export_onnx(model, path):
    import onnx
    import torch
    dummy = torch.zeros(1, 1, N_MELS, 301)
    kw = dict(input_names=["logmel"], output_names=["logits"])
    try:
        torch.onnx.export(model, (dummy,), str(path), opset_version=17, dynamo=False, **kw)
    except Exception as e:                                   # newer torch: dynamo exporter
        print(f"  (classic exporter unavailable: {type(e).__name__}; using dynamo)")
        torch.onnx.export(model, (dummy,), str(path), opset_version=18, dynamo=True, **kw)
    m = onnx.load(str(path))                                 # merge weights into a single file
    onnx.save_model(m, str(path), save_as_external_data=False)
    for extra in path.parent.glob(path.name + ".data"):
        extra.unlink()


def check_onnx(model, path):
    import onnxruntime as ort
    import torch
    x = torch.randn(2, 1, N_MELS, 301)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    errs = []
    for i in range(2):
        xi = x[i:i + 1]
        with torch.no_grad():
            ref = model(xi).numpy()
        out = sess.run(None, {sess.get_inputs()[0].name: xi.numpy()})[0]
        errs.append(np.abs(ref - out).max())
    print(f"  ONNX vs PyTorch: max error {max(errs):.2e}")
    assert max(errs) < 1e-3, "ONNX output differs from PyTorch"


def main():
    import librosa
    import torch
    OUT.mkdir(parents=True, exist_ok=True)

    fb = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS).astype(np.float64)   # (64, 257)
    print("Checking the browser preprocessing against librosa...")
    check_dsp(fb)

    entries = []
    for spec in MODELS:
        pt = Path(spec["pt"])
        if not pt.exists():
            print(f"- {spec['id']}: {pt} not found, skipped")
            continue
        print(f"- {spec['id']}: exporting {pt}")
        ckpt = torch.load(pt, map_location="cpu")
        classes = list(ckpt["classes"])
        model, arch = build_model(ckpt["model"], len(classes))
        path = OUT / f"{spec['id']}.onnx"
        export_onnx(model, path)
        check_onnx(model, path)

        uar = uar_std = None
        mfile = Path(spec.get("metrics") or "")
        if spec.get("metrics") and mfile.exists():
            m = json.loads(mfile.read_text()).get(spec["key"], {})
            uar, uar_std = m.get("mean_uar"), m.get("std_uar")
        entries.append(dict(id=spec["id"], file=path.name, arch=arch, corpus=spec["corpus"],
                            lang=spec["lang"], classes=classes, mu=float(ckpt["mu"]),
                            sd=float(ckpt["sd"]), uar=uar, uar_std=uar_std,
                            size_mb=round(path.stat().st_size / 1e6, 1)))

    if not entries:
        sys.exit("No model exported: check the paths in MODELS.")
    manifest = dict(sr=SR, n_samples=N_SAMPLES, n_fft=N_FFT, win=WIN, hop=HOP, n_mels=N_MELS,
                    trim_db=TRIM_DB, models=entries,
                    mel_fb=[[round(float(v), 8) for v in row] for row in fb])
    (OUT / "manifest.json").write_text(json.dumps(manifest))
    print(f"\nDone: {len(entries)} model(s) in {OUT}/ + manifest.json")


if __name__ == "__main__":
    main()
