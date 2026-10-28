# -*- coding: utf-8 -*-
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import os
import torchvision.transforms as transforms

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)

device = torch.device('cpu')
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load('ann_model.pth', map_location=device))
model.eval()

img_bgr = cv2.imread('./my_digits/1.png')
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
h, w = img_gray.shape

_, thresh = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
dilated = cv2.dilate(thresh, kernel, iterations=2)

boxes = []
for c in contours:
    x, y, bw, bh = cv2.boundingRect(c)
    if bw * bh > 500 and bw > 10 and bh > 20:
        boxes.append((x, y, bw, bh))

row1 = sorted([b for b in boxes if b[1] < h * 0.5], key=lambda b: b[0])
row2 = sorted([b for b in boxes if b[1] >= h * 0.5], key=lambda b: b[0])
sorted_boxes = row1 + row2

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

print("=" * 45)
print("{:<6} {:^10} {:^10} {:^10}".format("Sıra", "Beklenen", "Tahmin", "Güven"))
print("=" * 45)

predictions = []
confidences = []
correct_count = 0

for i, (x, y, bw, bh) in enumerate(sorted_boxes):
    roi = dilated[y:y+bh, x:x+bw]
    pad = abs(bh - bw) // 2
    if bh > bw:
        roi = cv2.copyMakeBorder(roi, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
    else:
        roi = cv2.copyMakeBorder(roi, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)

    roi_20 = cv2.resize(roi, (20, 20), interpolation=cv2.INTER_AREA)
    roi_28 = cv2.copyMakeBorder(roi_20, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)

    roi_pil = Image.fromarray(roi_28).convert('L')
    tensor = transform(roi_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)
        conf, pred = torch.max(probs, 1)

    p = pred.item()
    c = conf.item() * 100
    predictions.append(p)
    confidences.append(c)

    beklenen = str(i)
    is_ok = "[DOGRU]" if str(p) == beklenen else "[YANLIS]"
    if str(p) == beklenen:
        correct_count += 1
    print("{:<6} {:^10} {:^10} %{:<8.1f} {}".format(i+1, beklenen, str(p), c, is_ok))

print("=" * 45)
print("Toplam Basari: {}/{} (%{:.0f})".format(correct_count, len(sorted_boxes), 100*correct_count/len(sorted_boxes)))

for i, (x, y, bw, bh) in enumerate(sorted_boxes):
    beklenen = str(i)
    color = (0, 255, 0) if str(predictions[i]) == beklenen else (0, 0, 255)
    cv2.rectangle(img_bgr, (x, y), (x+bw, y+bh), color, 3)
    cv2.putText(img_bgr, str(predictions[i]), (x+5, y+45), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)

cv2.imwrite('./my_digits/sonuc.jpg', img_bgr)
print("\nsonuc.jpg kaydedildi!")
