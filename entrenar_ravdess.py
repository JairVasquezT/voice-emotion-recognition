#!/usr/bin/env python3
"""Train the scratch CNN on RAVDESS (English) with the same method as EmotionDetection.ipynb.

    python entrenar_ravdess.py --wav-dir DB/ravdess/wav            # cross-validation + final model
    python entrenar_ravdess.py --wav-dir DB/ravdess/wav --solo-final

Same features (16 kHz, central 3 s, 64-band log-mel -> 64 x 301), same ScratchCNN, same
training (Adam, cosine, 40 epochs, class weights, time shift + SpecAugment) as the EmoDB
notebook. Speaker-independent CV: 6 folds of 4 actors (2 male + 2 female).
"calm" is merged into "neutral" so the 7 classes match the face model (FER-2013).

Writes (paths expected by export_onnx.py):
    models/ravdess/scratch.pt        {"model", "mu", "sd", "classes", "corpus"}
    results/ravdess/metrics.json     {"scratch": {"fold_uar", "mean_uar", "std_uar"}}
    results/ravdess/scratch.png      confusion matrix (all folds pooled)

RAVDESS: Livingstone & Russo (2018), PLoS ONE 13(5): e0196391, CC BY-NC-SA 4.0.
File name: modality-channel-EMOTION-intensity-statement-repetition-ACTOR.wav (odd actor = male).
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import librosa
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, ConfusionMatrixDisplay

SR, N_SAMPLES, WIN, HOP, N_FFT, N_MELS = 16000, 3 * 16000, 400, 160, 512, 64
EMOTIONS = {"01": "neutral", "02": "neutral", "03": "happy", "04": "sad", "05": "angry",
            "06": "fearful", "07": "disgust", "08": "surprised"}      # 02 = calm -> neutral
CLASSES = sorted(set(EMOTIONS.values()))
FOLDS = [tuple("%02d" % a for a in range(k * 4 + 1, k * 4 + 5)) for k in range(6)]
EPOCHS = 40
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


# ----- features: identical to EmotionDetection.ipynb -----
def fix_length(x, n=N_SAMPLES):
    if len(x) < n:
        return np.pad(x, (0, n - len(x)))
    s = (len(x) - n) // 2
    return x[s:s + n]


def wav_to_logmel(x):
    x = x.astype(np.float32)
    x = x / (np.abs(x).max() + 1e-9)
    x = fix_length(x)
    S = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=N_FFT, win_length=WIN,
                                       hop_length=HOP, n_mels=N_MELS)
    return librosa.power_to_db(S, ref=1.0).astype(np.float32)   # (64, 301)


def load_features(wav_dir, cache):
    if os.path.exists(cache):
        d = np.load(cache)
        return d["X"], d["y"], d["spk"]
    X, y, spk = [], [], []
    paths = sorted(glob.glob(os.path.join(wav_dir, "**", "*.wav"), recursive=True))
    for i, path in enumerate(paths):
        parts = os.path.basename(path)[:-4].split("-")
        if len(parts) != 7 or parts[1] != "01":         # speech only
            continue
        x, _ = librosa.load(path, sr=SR, mono=True)
        X.append(wav_to_logmel(x)); y.append(CLASSES.index(EMOTIONS[parts[2]])); spk.append(parts[6])
        if (i + 1) % 200 == 0:
            print("  features %d/%d" % (i + 1, len(paths)))
    X, y, spk = np.stack(X), np.array(y), np.array(spk)
    np.savez(cache, X=X, y=y, spk=spk)
    return X, y, spk


# ----- model and training: identical to EmotionDetection.ipynb -----
def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2))


class ScratchCNN(nn.Module):
    def __init__(self, n_classes=len(CLASSES)):
        super().__init__()
        self.features = nn.Sequential(conv_block(1, 32), conv_block(32, 64),
                                      conv_block(64, 128), conv_block(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.3), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def augment(xb, max_shift=30, max_f=8, max_t=30):
    xb = xb.clone()
    n_mels, n_frames = xb.shape[-2:]
    for i in range(len(xb)):
        xb[i] = torch.roll(xb[i], np.random.randint(-max_shift, max_shift + 1), dims=-1)
        f = np.random.randint(0, max_f + 1); f0 = np.random.randint(0, n_mels - f + 1)
        t = np.random.randint(0, max_t + 1); t0 = np.random.randint(0, n_frames - t + 1)
        xb[i, :, f0:f0 + f, :] = 0
        xb[i, :, :, t0:t0 + t] = 0
    return xb


def fit(Xtr, ytr, epochs=EPOCHS, bs=32, lr=1e-3):
    Xtr, ytr = torch.tensor(Xtr).unsqueeze(1), torch.tensor(ytr)
    model = ScratchCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    counts = np.bincount(ytr.numpy(), minlength=len(CLASSES))
    w = torch.tensor(len(ytr) / (len(CLASSES) * np.maximum(counts, 1)), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=w.to(device))
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(ytr)); total = 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            xb, yb = augment(Xtr[idx]).to(device), ytr[idx].to(device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(idx)
        sched.step()
        if (ep + 1) % 10 == 0:
            print("    epoch %d/%d  loss %.3f" % (ep + 1, epochs, total / len(ytr)))
    return model


def predict(model, X, bs=64):
    model.eval(); preds = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            preds.append(model(torch.tensor(X[i:i + bs]).unsqueeze(1).to(device)).argmax(1).cpu())
    return torch.cat(preds).numpy()


def main():
    p = argparse.ArgumentParser(description="Train the scratch CNN on RAVDESS")
    p.add_argument("--wav-dir", default="DB/ravdess/wav")
    p.add_argument("--solo-final", action="store_true", help="skip cross-validation")
    args = p.parse_args()
    print("Device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    os.makedirs("models/ravdess", exist_ok=True)
    os.makedirs("results/ravdess", exist_ok=True)

    X, y, spk = load_features(args.wav_dir, "DB/ravdess_logmel.npz")
    print(X.shape, dict(zip(CLASSES, np.bincount(y).tolist())), "actors:", len(set(spk)))

    if not args.solo_final:
        t0 = time.time()
        all_pred, fold_uar = np.zeros_like(y), []
        for k, test_spk in enumerate(FOLDS):
            set_seed(k)
            te = np.isin(spk, test_spk); tr = ~te
            mu, sd = X[tr].mean(), X[tr].std()       # training speakers only
            model = fit((X[tr] - mu) / sd, y[tr])
            all_pred[te] = predict(model, (X[te] - mu) / sd)
            fold_uar.append(float(balanced_accuracy_score(y[te], all_pred[te])))
            print("  fold %d %s: UAR %.3f  (%.0f s)" % (k + 1, "+".join(test_spk), fold_uar[-1], time.time() - t0))
        metrics = {"scratch": {"fold_uar": [round(u, 3) for u in fold_uar],
                               "mean_uar": round(float(np.mean(fold_uar)), 4),
                               "std_uar": round(float(np.std(fold_uar)), 4)}}
        print("==> scratch: UAR %.3f +- %.3f (chance %.3f)" % (np.mean(fold_uar), np.std(fold_uar), 1 / len(CLASSES)))
        with open("results/ravdess/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay.from_predictions(y, all_pred, display_labels=CLASSES, normalize="true",
                                                values_format=".2f", xticks_rotation=45,
                                                cmap="Blues", ax=ax, colorbar=False)
        ax.set_title("RAVDESS scratch: UAR %.1f%% (unseen speakers)" % (100 * np.mean(fold_uar)))
        fig.tight_layout(); fig.savefig("results/ravdess/scratch.png", dpi=150)

    # Final model on all 24 actors (what the app and the robot use)
    set_seed(0)
    mu, sd = float(X.mean()), float(X.std())
    model = fit((X - mu) / sd, y)
    torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                "mu": mu, "sd": sd, "classes": CLASSES, "corpus": "RAVDESS"},
               "models/ravdess/scratch.pt")
    print("saved models/ravdess/scratch.pt")


if __name__ == "__main__":
    main()
