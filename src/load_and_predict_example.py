
"""Reload any saved model:  python src/load_and_predict_example.py models/ResNet18.pt some_crop.jpg  (run from the repo root)"""
import sys, torch, numpy as np
from PIL import Image
from coral_model_defs import build_model, CLASS_NAMES
ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
model = build_model(ck['name'], pretrained=False, mean=ck['data_mean'], std=ck['data_std'])
model.load_state_dict(ck['state_dict']); model.eval()
img = Image.open(sys.argv[2]).convert('RGB').resize((224, 224))                    # a 224x224 crop centred on the point of interest
x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float().div(255).unsqueeze(0)  # RGB in [0,1] - normalisation is inside the model
with torch.no_grad(): p = torch.softmax(model(x), 1)[0]
print({c: round(float(v), 4) for c, v in zip(CLASS_NAMES, p)})
