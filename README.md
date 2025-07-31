
# Captcha Solver (AI Technical Test)

## Problem

The task is to recognize 5-character captchas with consistent font, spacing, and no skew.

## Approach

1. **Segmentation:** Characters are equally spaced; we split the image into 5 parts.
2. **Training:** We train a CNN classifier on 28x28 grayscale character images labeled A-Z and 0-9.
3. **Inference:** Each character is predicted using the trained model and concatenated.

## Folder Structure

```
captcha_solver/
├── captcha.py              # Inference logic
├── train_model.py          # CNN training
├── dataset/train/          # Character images per class
├── models/
│   └── captcha_character_model.h5
└── README.md
```

## How to Use

```bash
python3 -m captcha.py /path/to/input.jpg /path/to/output.txt
```
