"""
Voice emotion recognition — web app (Gradio), EN / FR / ES.

Flow: record/upload audio -> log-mel (same preprocessing as the notebook)
      -> CNN from scratch + ResNet-18 -> probability per emotion.

Expected layout (the one the notebook produces):
    app.py
    models/scratch.pt
    models/resnet18.pt
    results/metrics.json   (optional, shows each model's UAR)
"""
import json
from pathlib import Path

import gradio as gr
import librosa
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
import numpy as np
import torch
import torch.nn as nn
import torchvision

ROOT = Path(__file__).parent
torch.set_num_threads(2)
DEFAULT_LANG = "en"

# ----------------------------------------------------------------------------
# 0. Texts in each language
# ----------------------------------------------------------------------------
T = {
    "en": {
        "intro": "# 🎙️ What emotion is in your voice?\n"
                 "Record a 2–5 second sentence **said with feeling** (anger, happiness, sadness, "
                 "fear, disgust, boredom or neutral) and press **Analyze**. You will see the "
                 "signal, its spectrogram and what each network predicts.\n\n"
                 "*Models trained on EmoDB (German actors recorded in a studio) and evaluated with "
                 "speaker-independent cross-validation. With a different microphone or language, "
                 "real-world accuracy is lower.*",
        "audio": "Your voice", "button": "Analyze", "plot": "Signal and spectrogram",
        "empty": "Record or upload audio to analyze it.",
        "too_short": "⚠️ The audio is too short or silent. Record at least 1–2 s of speech.",
        "duration": "Useful duration: {:.1f} s", "window_note": " → the central 3 s were analyzed.",
        "wave_title": "Speech signal", "window": "analyzed window (3 s)", "time": "time (s)",
        "spec_title": "Log-mel spectrogram (64 × 301): the image the CNN sees", "mel": "mel band",
        "models": {"scratch": "CNN from scratch", "resnet18": "ResNet-18 (transfer learning)"},
        "emotions": {"anger": "Anger", "boredom": "Boredom", "disgust": "Disgust", "fear": "Fear",
                     "happiness": "Happiness", "neutral": "Neutral", "sadness": "Sadness"},
    },
    "fr": {
        "intro": "# 🎙️ Quelle émotion y a-t-il dans votre voix ?\n"
                 "Enregistrez une phrase de 2 à 5 secondes **en y mettant de l'émotion** (colère, "
                 "joie, tristesse, peur, dégoût, ennui ou neutre) puis cliquez sur **Analyser**. "
                 "Vous verrez le signal, son spectrogramme et la prédiction de chaque réseau.\n\n"
                 "*Modèles entraînés sur EmoDB (acteurs allemands enregistrés en studio) et évalués "
                 "par validation croisée indépendante du locuteur. Avec un autre micro ou une autre "
                 "langue, la précision réelle est plus faible.*",
        "audio": "Votre voix", "button": "Analyser", "plot": "Signal et spectrogramme",
        "empty": "Enregistrez ou importez un audio pour l'analyser.",
        "too_short": "⚠️ L'audio est trop court ou silencieux. Enregistrez au moins 1 à 2 s de parole.",
        "duration": "Durée utile : {:.1f} s", "window_note": " → les 3 s centrales ont été analysées.",
        "wave_title": "Signal de parole", "window": "fenêtre analysée (3 s)", "time": "temps (s)",
        "spec_title": "Spectrogramme log-mel (64 × 301) : l'image que voit le CNN", "mel": "bande mel",
        "models": {"scratch": "CNN entraîné de zéro", "resnet18": "ResNet-18 (apprentissage par transfert)"},
        "emotions": {"anger": "Colère", "boredom": "Ennui", "disgust": "Dégoût", "fear": "Peur",
                     "happiness": "Joie", "neutral": "Neutre", "sadness": "Tristesse"},
    },
    "es": {
        "intro": "# 🎙️ ¿Qué emoción hay en tu voz?\n"
                 "Graba una frase de 2 a 5 segundos **diciéndola con emoción** (enojo, alegría, "
                 "tristeza, miedo, asco, aburrimiento o neutral) y pulsa **Analizar**. Verás la "
                 "señal, su espectrograma y lo que predice cada red.\n\n"
                 "*Modelos entrenados con EmoDB (actores alemanes grabados en estudio) y evaluados "
                 "con validación cruzada por locutor. Con otro micrófono u otro idioma la precisión "
                 "real es menor.*",
        "audio": "Tu voz", "button": "Analizar", "plot": "Señal y espectrograma",
        "empty": "Graba o sube un audio para analizarlo.",
        "too_short": "⚠️ El audio es muy corto o está en silencio. Graba al menos 1–2 s de voz.",
        "duration": "Duración útil: {:.1f} s", "window_note": " → se analizaron los 3 s centrales.",
        "wave_title": "Señal de voz", "window": "ventana analizada (3 s)", "time": "tiempo (s)",
        "spec_title": "Espectrograma log-mel (64 × 301): la imagen que ve la CNN", "mel": "banda mel",
        "models": {"scratch": "CNN desde cero", "resnet18": "ResNet-18 (transfer learning)"},
        "emotions": {"anger": "Enojo", "boredom": "Aburrimiento", "disgust": "Asco", "fear": "Miedo",
                     "happiness": "Alegría", "neutral": "Neutral", "sadness": "Tristeza"},
    },
}
LANGS = [("English", "en"), ("Français", "fr"), ("Español", "es")]

# ----------------------------------------------------------------------------
# 1. Preprocessing: IDENTICAL to the notebook (do not change these values)
# ----------------------------------------------------------------------------
SR, N_SAMPLES, WIN, HOP, N_FFT, N_MELS = 16000, 3 * 16000, 400, 160, 512, 64


def fix_length(x, n=N_SAMPLES):
    if len(x) < n:
        return np.pad(x, (0, n - len(x)))
    s = (len(x) - n) // 2
    return x[s:s + n]


def wav_to_logmel(x):
    x = x.astype(np.float32)
    x = x / (np.abs(x).max() + 1e-9)            # peak normalization
    x = fix_length(x)
    S = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=N_FFT, win_length=WIN,
                                       hop_length=HOP, n_mels=N_MELS)
    return librosa.power_to_db(S, ref=1.0).astype(np.float32)   # (64, 301)


# ----------------------------------------------------------------------------
# 2. Architectures: same as the notebook (inference only)
# ----------------------------------------------------------------------------
def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2))


class ScratchCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(conv_block(1, 32), conv_block(32, 64),
                                      conv_block(64, 128), conv_block(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.3), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


class TransferResNet(nn.Module):
    def __init__(self, n_classes, head_dropout=False):
        super().__init__()
        self.net = torchvision.models.resnet18(weights=None)   # weights come from the .pt
        fc = nn.Linear(self.net.fc.in_features, n_classes)
        # supports both versions: plain fc (original) or Dropout + fc (layer4 unfrozen)
        self.net.fc = nn.Sequential(nn.Dropout(0.3), fc) if head_dropout else fc

    def forward(self, x):
        return self.net(x.repeat(1, 3, 1, 1))


def load_model(path):
    ckpt = torch.load(path, map_location="cpu")
    state, classes = ckpt["model"], list(ckpt["classes"])
    if any(k.startswith("features.") for k in state):
        model = ScratchCNN(len(classes))
    else:
        model = TransferResNet(len(classes), head_dropout="net.fc.1.weight" in state)
    model.load_state_dict(state)
    model.eval()
    return {"model": model, "mu": float(ckpt["mu"]), "sd": float(ckpt["sd"]), "classes": classes}


metrics = {}
if (ROOT / "results/metrics.json").exists():
    metrics = json.loads((ROOT / "results/metrics.json").read_text())

MODELS = {}                                     # key -> {"model", "mu", "sd", "classes", "uar"}
for key in ("scratch", "resnet18"):
    path = ROOT / f"models/{key}.pt"
    if path.exists():
        MODELS[key] = load_model(path)
        MODELS[key]["uar"] = metrics.get(key, {}).get("mean_uar")
if not MODELS:
    raise FileNotFoundError("No model found in models/ (scratch.pt or resnet18.pt).")


def model_header(key, lang):
    name, uar = T[lang]["models"][key], MODELS[key]["uar"]
    return f"{name} — UAR {uar:.1%}" if uar is not None else name


# ----------------------------------------------------------------------------
# 3. Inference + visualization
# ----------------------------------------------------------------------------
def make_figure(x, logmel, lang):
    """Full waveform (with the 3 s window the model uses) + log-mel spectrogram."""
    t_ = T[lang]
    fig = Figure(figsize=(10, 5.2), tight_layout=True)
    ax1, ax2 = fig.subplots(2, 1, gridspec_kw={"height_ratios": [1, 2]})

    t = np.arange(len(x)) / SR
    ax1.plot(t, x, lw=0.6, color="#555")
    if len(x) > N_SAMPLES:                      # central part that goes into the model
        s = (len(x) - N_SAMPLES) // 2 / SR
        ax1.axvspan(s, s + 3, color="#e8500f", alpha=0.15, label=t_["window"])
        ax1.legend(loc="upper right", fontsize=8)
    ax1.set_xlim(0, max(t[-1], 3))
    ax1.set_yticks([])
    ax1.set_xlabel(t_["time"])
    ax1.set_title(t_["wave_title"], fontsize=10, loc="left")

    im = ax2.imshow(logmel, origin="lower", aspect="auto", cmap="magma",
                    extent=[0, N_SAMPLES / SR, 0, N_MELS])
    ax2.set_xlabel(t_["time"])
    ax2.set_ylabel(t_["mel"])
    ax2.set_title(t_["spec_title"], fontsize=10, loc="left")
    fig.colorbar(im, ax=ax2, format="%+.0f dB", pad=0.01)
    return fig


@torch.no_grad()
def analyze(audio_path, lang):
    """Returns [plot, one gr.Label per model, info text], all in the chosen language."""
    t_ = T[lang]

    def result(fig=None, probs=None, info=""):
        return [gr.Plot(value=fig, label=t_["plot"]),
                *[gr.Label(value=(probs or {}).get(k), label=model_header(k, lang))
                  for k in MODELS],
                info]

    if audio_path is None:
        return result(info=t_["empty"])

    x, _ = librosa.load(audio_path, sr=SR, mono=True)        # resample to 16 kHz mono
    x, _ = librosa.effects.trim(x, top_db=30)               # remove leading/trailing silence
    if len(x) < 0.5 * SR or np.abs(x).max() < 1e-3:
        return result(info=t_["too_short"])

    logmel = wav_to_logmel(x)
    fig = make_figure(x / (np.abs(x).max() + 1e-9), logmel, lang)

    probs, votes = {}, []
    for key, m in MODELS.items():
        inp = torch.from_numpy((logmel - m["mu"]) / m["sd"])[None, None]   # (1, 1, 64, 301)
        p = torch.softmax(m["model"](inp), dim=1)[0].numpy()
        probs[key] = {t_["emotions"].get(c, c): float(v) for c, v in zip(m["classes"], p)}
        best = max(probs[key], key=probs[key].get)
        votes.append(f"**{t_['models'][key]}:** {best} ({p.max():.0%})")

    info = (t_["duration"].format(len(x) / SR)
            + (t_["window_note"] if len(x) > N_SAMPLES else ".")
            + "\n\n" + " · ".join(votes))
    return result(fig, probs, info)


def translate_ui(lang):
    t_ = T[lang]
    return gr.Markdown(t_["intro"]), gr.Audio(label=t_["audio"]), gr.Button(t_["button"])


# ----------------------------------------------------------------------------
# 4. Interface
# ----------------------------------------------------------------------------
t0 = T[DEFAULT_LANG]
with gr.Blocks(title="Voice emotion recognition") as demo:
    with gr.Row():
        with gr.Column(scale=4):
            intro = gr.Markdown(t0["intro"])
        with gr.Column(scale=1, min_width=200):
            lang = gr.Radio(LANGS, value=DEFAULT_LANG, label="🌐 Language / Langue / Idioma")
    with gr.Row():
        with gr.Column(scale=1):
            audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label=t0["audio"])
            btn = gr.Button(t0["button"], variant="primary")
            info = gr.Markdown()
        with gr.Column(scale=2):
            plot = gr.Plot(label=t0["plot"])
    with gr.Row():
        labels = [gr.Label(num_top_classes=7, label=model_header(k, DEFAULT_LANG)) for k in MODELS]

    outs = [plot, *labels, info]
    btn.click(analyze, inputs=[audio, lang], outputs=outs)
    audio.stop_recording(analyze, inputs=[audio, lang], outputs=outs)   # analyze when recording stops
    audio.upload(analyze, inputs=[audio, lang], outputs=outs)
    # changing the language translates the UI and re-labels the last result
    lang.change(translate_ui, inputs=lang, outputs=[intro, audio, btn]) \
        .then(analyze, inputs=[audio, lang], outputs=outs)

if __name__ == "__main__":
    demo.launch()
