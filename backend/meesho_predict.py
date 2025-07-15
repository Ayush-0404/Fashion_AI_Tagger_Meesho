import os
import json
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
import pandas as pd

# Load category attributes
cat_attrs = pd.read_parquet("category_attributes.parquet")

# category -> number of attributes
cat2N = dict(zip(cat_attrs.Category, cat_attrs.No_of_attribute))

# category -> slot index -> attribute name
category_slot_to_attr = {}
for _, row in cat_attrs.iterrows():
    cat = row["Category"]
    N = int(row["No_of_attribute"])
    attrs = list(row["Attribute_list"])
    slot_map = {j: attrs[j-1] for j in range(1, N+1)}
    category_slot_to_attr[cat] = slot_map

with open("category_slot_to_labels.json") as f:
    category_slot_to_labels = json.load(f)

category_slot_to_labels = {
    cat: {int(j): set(labels) for j, labels in slots.items()}
    for cat, slots in category_slot_to_labels.items()
}


# Load label2idx
with open("label2idx.json") as f:
    label2idx = json.load(f)

# Convert keys back to int
label2idx = {int(k): v for k,v in label2idx.items()}


# 3. Preprocessing
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# 4. Load Model and Weights 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Build model
backbone = torchvision.models.efficientnet_b0(
    weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT)
backbone.classifier = nn.Identity()
feature_dim = 1280

heads = nn.ModuleList([
    nn.Linear(feature_dim, len(label2idx[j]))
    for j in range(1,11)
])

# Load weights
ckpt = torch.load("best_model.pth", map_location=device, weights_only=False)
backbone.load_state_dict(ckpt["backbone"])
heads.load_state_dict(ckpt["heads"])

backbone.to(device).eval()
heads.to(device).eval()



# 5. Category Inference Helper

def infer_category_from_preds(pred_labels, cat2N, category_slot_to_labels):
    scores = {}
    for cat, N in cat2N.items():
        count = 0
        for j in range(1,N+1):
            pred_label = pred_labels[j-1]
            if pred_label in category_slot_to_labels[cat][j]:
                count += 1
        scores[cat] = count
    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
    best_cat = sorted_scores[0][0]
    return best_cat

def predict_folder(folder_path):
    results = {}

    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(('.jpg','.jpeg','.png'))]

    with torch.no_grad():
        for fname in image_files:
            img_path = os.path.join(folder_path, fname)

            # Load and preprocess
            img = Image.open(img_path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)

            # Extract features
            feats = backbone(x)

            # Predict labels for all 10 slots
            pred_labels = []
            for j in range(1,11):
                logits = heads[j-1](feats)
                pred_idx = logits.argmax(dim=1).item()
                label = [k for k,v in label2idx[j].items() if v==pred_idx][0]
                pred_labels.append(label)

            # Infer category
            pred_cat = infer_category_from_preds(pred_labels, cat2N, category_slot_to_labels)

            # Build dict of relevant attributes
            N = cat2N[pred_cat]
            attrs = {}
            for j in range(1,N+1):
                attr_name = category_slot_to_attr[pred_cat][j]
                attrs[attr_name] = pred_labels[j-1]

            results[fname] = {
                "category": pred_cat,
                "attributes": attrs
            }

    return results

def predict_attributes(image: Image.Image) -> dict:
    with torch.no_grad():
        x = transform(image).unsqueeze(0).to(device)
        feats = backbone(x)
        pred_labels = []
        for j in range(1, 11):
            logits = heads[j-1](feats)
            pred_idx = logits.argmax(dim=1).item()
            label = [k for k, v in label2idx[j].items() if v == pred_idx][0]
            pred_labels.append(label)
        pred_cat = infer_category_from_preds(pred_labels, cat2N, category_slot_to_labels)
        N = cat2N[pred_cat]
        attrs = {}
        for j in range(1, N+1):
            attr_name = category_slot_to_attr[pred_cat][j]
            attrs[attr_name] = pred_labels[j-1]
        return {
            "category": pred_cat,
            "attributes": attrs
        }

if __name__ == "__main__":
    # Example folder path
    folder = "images"

    output = predict_folder(folder)

    # Save as JSON
    with open("predictions.json", "w") as f:
        json.dump(output, f, indent=2)

    print("Done. Predictions saved to predictions.json")