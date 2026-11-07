"""
Command-line inference with the trained coral-reef health models.

Examples
--------
# whole quadrat -> dense health map + % cover + reef status
python src/predict.py --image my_quadrat.jpg --model DINOv2-ViT-S/14 --out outputs/

# classify specific points (CSV with columns Row,Column) on a quadrat
python src/predict.py --image my_quadrat.jpg --points points.csv --model ResNet18 --out outputs/

# classify a single 224x224 crop centred on the point of interest
python src/predict.py --crop crop.jpg --model CoralCNN-SE
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coral_model_defs import CLASS_NAMES  # noqa: E402
from inference import MODEL_FILES, analyse_quadrat, apply_underwater_filter, load_model, point_crops, predict_batch  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="DINOv2-ViT-S/14", choices=list(MODEL_FILES))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="whole quadrat image")
    g.add_argument("--crop", help="single crop centred on the point of interest (resized to 224x224)")
    ap.add_argument("--points", help="CSV with Row,Column pixel coordinates to classify (requires --image)")
    ap.add_argument("--stride", type=int, default=112, help="sliding-window stride in px (224, 112 or 56)")
    ap.add_argument("--max-side", type=int, default=None, help="optionally downscale large images to this longest side")
    ap.add_argument("--no-filter", action="store_true", help="skip the gray-world + CLAHE colour correction")
    ap.add_argument("--tau", type=float, default=None, help="override the validation-tuned prior-correction strength")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--out", default="outputs", help="output folder")
    a = ap.parse_args()

    lm = load_model(a.model, device=a.device, tau=a.tau)
    print(f"Loaded {a.model} (tau = {lm.tau}) on {lm.device}")

    if a.crop:
        img = np.asarray(Image.open(a.crop).convert("RGB").resize((224, 224)))
        if not a.no_filter:
            img = apply_underwater_filter(img)
        p = predict_batch(lm, img[None])[0]
        print(json.dumps({c: round(float(v), 4) for c, v in zip(CLASS_NAMES, p)}, indent=2))
        return

    os.makedirs(a.out, exist_ok=True)
    rgb = np.asarray(Image.open(a.image).convert("RGB"))
    stem = os.path.splitext(os.path.basename(a.image))[0]

    if a.points:
        import pandas as pd
        pts = pd.read_csv(a.points)
        img = rgb if a.no_filter else apply_underwater_filter(rgb)
        probs = predict_batch(lm, point_crops(img, pts["Row"].values, pts["Column"].values))
        pts["Pred_class"] = [CLASS_NAMES[i] for i in probs.argmax(1)]
        for i, c in enumerate(CLASS_NAMES):
            pts[f"P_{c.split()[0]}"] = probs[:, i]
        path = os.path.join(a.out, f"{stem}_points.csv")
        pts.to_csv(path, index=False)
        print(pts["Pred_class"].value_counts(normalize=True).mul(100).round(1).to_string())
        print("Saved", path)
        return

    res = analyse_quadrat(lm, rgb, stride=a.stride, max_side=a.max_side, filter_image=not a.no_filter)
    Image.fromarray(res["overlay"]).save(os.path.join(a.out, f"{stem}_health_map.png"))
    with open(os.path.join(a.out, f"{stem}_cover.json"), "w") as f:
        json.dump(dict(model=a.model, stride=a.stride, **res["cover"]), f, indent=2)
    print(json.dumps(res["cover"], indent=2, ensure_ascii=False))
    print("Saved overlay and cover summary to", a.out)


if __name__ == "__main__":
    main()
