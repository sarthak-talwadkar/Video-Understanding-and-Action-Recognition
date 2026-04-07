# Video Action Recognition — Voxel51 Hackathon

> **Built at the Voxel51 Video Understanding AI Hackathon @ Northeastern University — April 3, 2026**
> Team of 4 · One-day build · 60% clip-level classification accuracy on Kinetics

**Clip-level video action recognition pipeline** built on the FiftyOne ecosystem — given a video, identify what actions are happening and when. Combines **X-CLIP** (cross-modal video-language alignment) and **VideoMAE** (masked autoencoder video representations) with a custom FiftyOne plugin for interactive visualization and agent-driven analysis of action predictions across a Kinetics dataset.

---

## Demo

![Action Recognition Demo](assets/demo.gif)

> FiftyOne plugin panel: video clips with predicted action labels overlaid, temporal segments highlighted, confidence scores displayed per clip. X-CLIP and VideoMAE predictions compared side-by-side.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Models](#models)
  - [X-CLIP — Cross-Modal Video-Language Alignment](#x-clip--cross-modal-video-language-alignment)
  - [VideoMAE — Masked Autoencoder for Video](#videomae--masked-autoencoder-for-video)
  - [CLIP — Zero-Shot Action Grounding](#clip--zero-shot-action-grounding)
  - [Model Comparison](#model-comparison)
- [FiftyOne Integration](#fiftyone-integration)
  - [Dataset Zoo — Kinetics](#dataset-zoo--kinetics)
  - [Custom Plugin](#custom-plugin)
  - [FiftyOne Agent](#fiftyone-agent)
  - [Model Zoo](#model-zoo)
- [Video Preprocessing Pipeline](#video-preprocessing-pipeline)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Team](#team)
- [References](#references)

---

## Overview

Video action recognition is the task of assigning semantic action labels to video clips — understanding not just *what* is in a frame but *what is happening* over time. This is significantly harder than image classification because:

- **Temporal reasoning is required** — "throwing a ball" and "catching a ball" look nearly identical in any single frame
- **Action boundaries are ambiguous** — actions blend into each other without hard cuts
- **Appearance alone is insufficient** — motion patterns, object interactions, and temporal context are all necessary signals

This project builds a full clip-level action recognition pipeline in one day using two complementary approaches:

- **X-CLIP:** Leverages language supervision — action labels are encoded as text and matched against video clip embeddings, enabling zero-shot generalization to unseen action categories
- **VideoMAE:** Learns rich spatiotemporal features via self-supervised masked autoencoding, fine-tuned for action classification on Kinetics

Both models are integrated into a **custom FiftyOne plugin** that provides interactive visualization, side-by-side model comparison, and agent-driven dataset querying.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Input Video                             │
│              (Kinetics clips via FiftyOne Zoo)              │
└──────────────────────────┬──────────────────────────────────┘
                           │
             ┌─────────────▼──────────────┐
             │   Video Preprocessing      │
             │   Temporal clip sampling   │
             │   Frame extraction         │
             │   Resize + normalize       │
             └──────┬──────────┬──────────┘
                    │          │
       ┌────────────▼──┐   ┌───▼────────────────┐
       │    X-CLIP     │   │     VideoMAE        │
       │  Video+Text   │   │  Spatiotemporal     │
       │  Encoder      │   │  Masked Autoencoder │
       │               │   │                     │
       │ video embed   │   │  video features     │
       │ text embed    │   │  → classifier head  │
       │ cosine sim    │   │                     │
       │ → action label│   │  → action label     │
       └────────┬──────┘   └──────────┬──────────┘
                │                     │
                └──────────┬──────────┘
                           │ Predictions + confidence
                           ▼
             ┌─────────────────────────────┐
             │     FiftyOne Plugin         │
             │  Custom operator panel      │
             │  Prediction visualization  │
             │  Model comparison view      │
             │  Agent-driven querying      │
             └─────────────────────────────┘
```

---

## Models

### X-CLIP — Cross-Modal Video-Language Alignment

X-CLIP (Ni et al., 2022) extends CLIP to video by adding a **cross-frame attention mechanism** and a **video-specific prompting module** that generates frame-conditioned text representations. It enables zero-shot action recognition — no fine-tuning on the target action vocabulary is required.

**Architecture:**

```
Video Input (T frames)
    │
    ▼
┌───────────────────────────────────────────┐
│  Frame-level Visual Encoder               │
│  (CLIP ViT-B/16, applied per frame)       │
│  → T frame embeddings ∈ R^{T×512}        │
└───────────────────┬───────────────────────┘
                    │
┌───────────────────▼───────────────────────┐
│  Cross-Frame Attention                    │
│  Each frame attends to all other frames   │
│  → temporally aggregated video embedding  │
│  v ∈ R^512                               │
└───────────────────┬───────────────────────┘
                    │
                    │        Action label text
                    │              │
                    │    ┌─────────▼──────────┐
                    │    │  Video-Conditioned  │
                    │    │  Text Prompting     │
                    │    │  t_i = f(label_i,  │
                    │    │         v)          │
                    │    │  → t_i ∈ R^512     │
                    │    └─────────┬──────────┘
                    │              │
                    └──────┬───────┘
                           ▼
              similarity(v, t_i) = v·t_i / (||v||·||t_i||)
              → softmax over all action labels
              → predicted action
```

**Video-conditioned text prompting** is X-CLIP's key innovation over applying CLIP directly to video. Standard CLIP text embeddings are static — "a person running" means the same thing regardless of what's in the video. X-CLIP conditions the text embedding on the video content, producing prompts like "a person running [on a track, in slow motion, from behind]" implicitly — making the text-video similarity more discriminative.

```python
from transformers import XCLIPProcessor, XCLIPModel
import torch

model = XCLIPModel.from_pretrained("microsoft/xclip-base-patch32")
processor = XCLIPProcessor.from_pretrained("microsoft/xclip-base-patch32")

# Action label set (Kinetics-400 classes)
action_labels = ["running", "jumping", "swimming", "cooking", ...]

# Process video frames + text labels
inputs = processor(
    text=action_labels,
    videos=list(video_frames),   # list of T PIL images
    return_tensors="pt",
    padding=True
)

with torch.no_grad():
    outputs = model(**inputs)

# Cosine similarity between video and each action label
probs = outputs.logits_per_video.softmax(dim=-1)
predicted_action = action_labels[probs.argmax()]
```

**Zero-shot capability:** X-CLIP can predict action labels never seen during training — as long as the label can be expressed as natural language, the shared video-text embedding space provides a similarity score.

---

### VideoMAE — Masked Autoencoder for Video

VideoMAE (Tong et al., 2022) adapts the Masked Autoencoder (MAE) framework to video. During pretraining, **90–95% of video patches are randomly masked** and the model learns to reconstruct the missing content — forcing it to learn rich spatiotemporal representations to fill in masked regions from sparse context.

**Why such high masking ratio?**

Video has high temporal redundancy — adjacent frames are nearly identical. Masking 75% (as in image MAE) is too easy; the model can reconstruct missing patches by copying from neighboring frames without learning anything meaningful. Masking 90–95% forces the model to reason about motion, object persistence, and temporal dynamics to reconstruct the missing content.

**Architecture:**

```
Video Input: T×H×W×3
    │
    ▼
Tubelet Embedding
(2×16×16 spatiotemporal patches → tokens)
    │
    ▼
Random Masking (90% of tokens removed)
    │
    ▼
┌───────────────────────────────┐
│  ViT Encoder                  │
│  (operates on visible tokens  │
│   only — computationally      │
│   efficient)                  │
└───────────────┬───────────────┘
                │ encoded visible tokens
                ▼
┌───────────────────────────────┐
│  Decoder                      │
│  (lightweight transformer)    │
│  Reconstruct masked patches   │
│  from visible context         │
└───────────────┬───────────────┘
                │
           Reconstruction
           loss (MSE on
           pixel values)
```

**Tubelet embedding:** Rather than treating each frame independently, VideoMAE uses 3D patch tokens (2 frames × 16×16 pixels). This captures short-range temporal motion within each token, making the representation inherently spatiotemporal.

**Fine-tuning for action classification:**

The pretrained encoder is used as a feature extractor, with a classification head trained on Kinetics:

```python
from transformers import VideoMAEForVideoClassification, VideoMAEFeatureExtractor

feature_extractor = VideoMAEFeatureExtractor.from_pretrained(
    "MCG-NJU/videomae-base-finetuned-kinetics"
)
model = VideoMAEForVideoClassification.from_pretrained(
    "MCG-NJU/videomae-base-finetuned-kinetics"
)

# Sample 16 frames uniformly from clip
inputs = feature_extractor(video_frames, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

predicted_class = model.config.id2label[outputs.logits.argmax(-1).item()]
```

---

### CLIP — Zero-Shot Action Grounding

Beyond X-CLIP, raw CLIP (ViT-B/32) is used for **frame-level action grounding** — identifying which frames within a clip are most semantically aligned with a given action label. This provides temporal localization: rather than just classifying the whole clip, CLIP scores each frame against the action label and highlights the temporal segment where the action is most active.

```python
import clip
import torch
import numpy as np

model, preprocess = clip.load("ViT-B/32", device="cuda")

def localize_action(frames, action_label):
    """
    Returns per-frame similarity scores for temporal localization.
    frames: list of PIL images (one per sampled frame)
    action_label: string e.g. "playing tennis"
    """
    text = clip.tokenize([action_label]).cuda()
    with torch.no_grad():
        text_feat = model.encode_text(text)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)

    scores = []
    for frame in frames:
        img = preprocess(frame).unsqueeze(0).cuda()
        with torch.no_grad():
            img_feat = model.encode_image(img)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
        scores.append((img_feat * text_feat).sum().item())

    return np.array(scores)   # shape: (T,) — per-frame similarity

# Smooth scores and find peak temporal segment
from scipy.ndimage import gaussian_filter1d
smoothed = gaussian_filter1d(scores, sigma=2)
peak_frame = np.argmax(smoothed)
```

The smoothed per-frame CLIP similarity curve provides a lightweight temporal attention map — no temporal model required — useful for quickly identifying the most action-relevant segment of a long video.

---

### Model Comparison

| Property | CLIP (frame-level) | X-CLIP | VideoMAE |
|---|---|---|---|
| Temporal modeling | None — per frame | Cross-frame attention | Spatiotemporal tubelets |
| Training supervision | Image-text contrastive | Video-text contrastive | Self-supervised (reconstruction) |
| Zero-shot capability | Yes | Yes | No (requires fine-tuning) |
| Action vocabulary | Open (any text label) | Open (any text label) | Fixed (Kinetics classes) |
| Temporal localization | Yes (frame scores) | No (clip-level only) | No (clip-level only) |
| Inference speed | Fast | Medium | Slow (ViT encoder) |
| Accuracy on Kinetics | Lower | Medium | Higher |

---

## FiftyOne Integration

### Dataset Zoo — Kinetics

Kinetics-400 clips are loaded directly via FiftyOne's Dataset Zoo:

```python
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
    "kinetics-400",
    split="validation",
    max_samples=500,
    shuffle=True,
    seed=42,
)
```

Each sample in the dataset is a video clip with ground truth action label, enabling direct accuracy evaluation against predictions.

### Custom Plugin

A custom FiftyOne plugin was built during the hackathon providing:

- **Prediction panel:** Displays X-CLIP and VideoMAE predictions side-by-side with confidence scores per clip
- **Temporal localization overlay:** CLIP per-frame similarity scores visualized as a timeline heatmap below each video
- **Model comparison operator:** One-click toggle between X-CLIP, VideoMAE, and ensemble predictions
- **Filtering operator:** Filter dataset by predicted action, confidence threshold, or model agreement/disagreement

```python
import fiftyone.operators as foo
import fiftyone.operators.types as types

class ActionPredictionOperator(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="predict_actions",
            label="Predict Actions",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.enum(
            "model",
            values=["xclip", "videomae", "ensemble"],
            label="Model",
            default="xclip",
        )
        inputs.float(
            "confidence_threshold",
            label="Confidence Threshold",
            default=0.3,
        )
        return types.Property(inputs)

    def execute(self, ctx):
        model_choice = ctx.params.get("model", "xclip")
        threshold = ctx.params.get("confidence_threshold", 0.3)

        for sample in ctx.dataset.iter_samples(autosave=True):
            frames = load_frames(sample.filepath)
            if model_choice == "xclip":
                label, conf = predict_xclip(frames)
            elif model_choice == "videomae":
                label, conf = predict_videomae(frames)
            else:
                label, conf = predict_ensemble(frames)

            if conf >= threshold:
                sample["predicted_action"] = fo.Classification(
                    label=label, confidence=conf
                )
```

### FiftyOne Agent

A FiftyOne agent was configured to answer natural language queries over the dataset using the MCP (Model Context Protocol) interface:

```
User: "Show me all clips where X-CLIP and VideoMAE disagree"
Agent: filters dataset → returns clips where predictions differ

User: "Which action categories have the lowest confidence?"
Agent: aggregates confidence by label → returns ranked list

User: "Find clips of 'juggling' that VideoMAE misclassified"
Agent: filters by ground truth + prediction mismatch
```

The agent wraps the FiftyOne Python SDK behind a natural language interface, enabling interactive dataset exploration without writing filter queries manually.

### Model Zoo

Pretrained video models were loaded via FiftyOne's Model Zoo for baseline comparison:

```python
model = foz.load_zoo_model("clip-vit-base32-torch")
dataset.apply_model(model, label_field="clip_predictions")
```

---

## Video Preprocessing Pipeline

All three models require different input formats — a shared preprocessing pipeline normalizes inputs before model-specific formatting:

```python
import cv2
import numpy as np
from PIL import Image

def preprocess_video(filepath, n_frames=16, target_size=(224, 224)):
    """
    Uniformly sample n_frames from video, resize, normalize.
    Returns: list of PIL images (for X-CLIP / CLIP)
             np.array of shape (T, H, W, 3) (for VideoMAE)
    """
    cap = cv2.VideoCapture(filepath)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Uniform temporal sampling
    indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, target_size)
        frames.append(Image.fromarray(frame))

    cap.release()
    return frames

def frames_to_array(frames):
    """Convert PIL frame list to float32 array normalized to [0,1]"""
    arr = np.stack([np.array(f) for f in frames])   # T×H×W×3
    return arr.astype(np.float32) / 255.0
```

**Uniform temporal sampling** (vs. dense or random sampling): ensures consistent coverage of the full clip regardless of video length, which is important for action recognition where the key motion may occur at any point in the clip.

---

## Results

| Model | Top-1 Accuracy | Notes |
|---|---|---|
| CLIP (frame-level, zero-shot) | ~35% | No temporal modeling — per-frame only |
| X-CLIP (zero-shot) | ~52% | Cross-frame attention, open vocabulary |
| VideoMAE (fine-tuned Kinetics) | **~60%** | Best accuracy, fixed Kinetics vocabulary |
| Ensemble (X-CLIP + VideoMAE) | ~58% | Averaged logits — marginal improvement |

**60% top-1 accuracy** on Kinetics-400 validation subset (500 clips) within a single hackathon day using pretrained models with no additional fine-tuning on the evaluation subset.

### Key Observations

- **VideoMAE outperforms X-CLIP** despite X-CLIP's richer language supervision — supervised fine-tuning on Kinetics gives VideoMAE a distribution-matched classifier, while X-CLIP's zero-shot alignment incurs a vocabulary mismatch penalty
- **CLIP frame-level baseline is surprisingly competitive** for static actions (e.g., "cooking", "reading") but fails on motion-dependent actions (e.g., "running", "swimming") — confirming that temporal modeling is necessary for motion understanding
- **Ensemble underperforms VideoMAE alone** — X-CLIP's zero-shot errors are not independent of VideoMAE's errors (they tend to fail on the same ambiguous clips), so averaging logits doesn't help
- **FiftyOne plugin** revealed that most errors cluster around visually similar action pairs (e.g., "playing tennis" vs "playing badminton", "surfing" vs "skateboarding") — indicating feature confusion at the clip embedding level rather than labeling noise

---

## Installation

```bash
git clone https://github.com/sarthak-talwadkar/video-action-recognition.git
cd video-action-recognition
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, PyTorch ≥ 1.13, transformers, fiftyone, clip (`pip install git+https://github.com/openai/CLIP.git`), OpenCV, scipy

**Install FiftyOne plugin:**
```bash
fiftyone plugins download .
```

---

## Usage

### Load Kinetics dataset and run predictions

```bash
python predict.py \
    --model xclip \
    --n-samples 500 \
    --output-field predicted_action
```

### Launch FiftyOne App with custom plugin

```bash
python app.py
# Opens FiftyOne App at localhost:5151
# Use "Predict Actions" operator in the plugin panel
```

### Run model comparison

```bash
python compare_models.py \
    --models xclip videomae ensemble \
    --n-samples 500 \
    --plot
```

---

## Project Structure

```
video-action-recognition/
├── models/
│   ├── xclip_predictor.py      # X-CLIP inference + action label scoring
│   ├── videomae_predictor.py   # VideoMAE inference + classification
│   ├── clip_localizer.py       # CLIP frame-level temporal localization
│   └── ensemble.py             # Ensemble prediction (averaged logits)
├── plugin/
│   ├── __init__.py
│   └── operator.py             # FiftyOne custom operator (ActionPredictionOperator)
├── data/
│   └── kinetics_loader.py      # FiftyOne Dataset Zoo loader + preprocessing
├── utils/
│   ├── preprocess.py           # Uniform frame sampling + normalization
│   └── visualize.py            # Temporal score curves, confusion matrix
├── predict.py                  # Run predictions on dataset
├── compare_models.py           # Multi-model benchmark + plots
├── app.py                      # Launch FiftyOne App with plugin
├── requirements.txt
└── README.md
```

---

## Team

Built in one day at the **Voxel51 Video Understanding AI Hackathon @ Northeastern University, April 3, 2026**.

| Member | Contribution |
|---|---|
| **Sarthak Talwadkar** | CLIP integration, X-CLIP + VideoMAE video model pipeline |
| Teammate 2 | *(add contribution)* |
| Teammate 3 | *(add contribution)* |
| Teammate 4 | *(add contribution)* |

---

## References

- Ni, B. et al. *"Expanding Language-Image Pretrained Models for General Video Recognition."* ECCV 2022. [[Paper]](https://arxiv.org/abs/2208.02816) *(X-CLIP)*
- Tong, Z. et al. *"VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training."* NeurIPS 2022. [[Paper]](https://arxiv.org/abs/2203.12602)
- Radford, A. et al. *"Learning Transferable Visual Models From Natural Language Supervision."* ICML 2021. [[Paper]](https://arxiv.org/abs/2103.00020) *(CLIP)*
- FiftyOne Documentation. [[Docs]](https://docs.voxel51.com)
- Kinetics-400 Dataset. [[Paper]](https://arxiv.org/abs/1705.06950)

---

## Author

**Sarthak Talwadkar**
MS Robotics, Northeastern University
[LinkedIn](https://linkedin.com/in/sarthak-talwadkar) · [GitHub](https://github.com/sarthak-talwadkar)
