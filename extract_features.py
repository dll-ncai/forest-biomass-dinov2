#!/usr/bin/env python3
"""
Extract frozen features from image crops for both study sites and both backbones.

Backbones (timm, ImageNet-pretrained, frozen, classification head removed):
  * dino     : vit_small_patch16_224.dino  -> 384-d CLS token (self-supervised ViT)
  * resnet50 : resnet50                     -> 2048-d global-avg-pool (supervised CNN)

Sites:
  * pakistan : dataset_crops_152/crop_<idx>_label_<kg>.png  (152x152)
               preprocessing: 152 -> center-crop 112 -> resize 224 (bicubic)
  * german   : german_data_results/complete_pipeline_dataset/crops/*.png (224x224)
               preprocessing: identity resize to 224

All crops are normalized with ImageNet mean/std, matching the original pipeline.
Output CSV columns: feature_0..feature_{D-1}, label, [split], source_file.
"""
import os, re, glob, argparse
import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image
import torchvision.transforms as T

ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MODEL_SPECS = {
    "dino":     dict(name="vit_small_patch16_224.dino", dim=384),
    "resnet50": dict(name="resnet50",                   dim=2048),
}


def build_model(kind, device):
    spec = MODEL_SPECS[kind]
    model = timm.create_model(spec["name"], pretrained=True, num_classes=0)
    model.eval().to(device)
    return model


def pakistan_transform():
    # 152 crop -> center 112 -> resize 224 bicubic -> ImageNet norm
    return T.Compose([
        T.CenterCrop(112),
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def german_transform():
    return T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_pakistan_items():
    items = []
    for f in glob.glob(os.path.join(ROOT, "dataset_crops_152", "crop_*_label_*.png")):
        m = re.search(r"crop_(\d+)_label_(\d+)", os.path.basename(f))
        items.append((int(m.group(1)), f, float(m.group(2))))
    items.sort()
    return [(f, lab) for _, f, lab in items]


def load_german_items():
    items = []
    for f in glob.glob(os.path.join(ROOT, "german_data_results",
                                    "complete_pipeline_dataset", "crops", "*.png")):
        m = re.search(r"biomass_([0-9.]+)\.png$", os.path.basename(f))
        if m:
            items.append((os.path.basename(f), f, float(m.group(1))))
    items.sort()
    return [(f, lab) for _, f, lab in items]


@torch.no_grad()
def extract(kind, site, out_csv, batch_size=16):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(kind, device)
    tf = pakistan_transform() if site == "pakistan" else german_transform()
    items = load_pakistan_items() if site == "pakistan" else load_german_items()
    print(f"[{site}/{kind}] {len(items)} crops on {device}")

    feats, labels, files = [], [], []
    batch, meta = [], []
    def flush():
        if not batch:
            return
        x = torch.stack(batch).to(device)
        y = model(x).cpu().numpy()
        feats.append(y)
        for fpath, lab in meta:
            labels.append(lab); files.append(os.path.basename(fpath))
        batch.clear(); meta.clear()

    for fpath, lab in items:
        img = Image.open(fpath).convert("RGB")
        batch.append(tf(img)); meta.append((fpath, lab))
        if len(batch) >= batch_size:
            flush()
    flush()

    F = np.concatenate(feats, axis=0)
    cols = [f"feature_{i}" for i in range(F.shape[1])]
    df = pd.DataFrame(F, columns=cols)
    df["label"] = labels
    df["source_file"] = files
    df.to_csv(out_csv, index=False)
    print(f"  -> wrote {out_csv}  shape={df.shape}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_SPECS), required=True)
    ap.add_argument("--site", choices=["pakistan", "german"], required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(ROOT, f"features_{args.site}_{args.model}.csv")
    extract(args.model, args.site, out)


if __name__ == "__main__":
    main()
