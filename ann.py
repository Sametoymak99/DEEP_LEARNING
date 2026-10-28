# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1. VERİ SETİ YÜKLEME VE ÖN İŞLEME
# ============================================================

def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_set = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loaders = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    test_loaders = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )

    return train_loaders, test_loaders

train_dataset, test_dataset = get_data_loaders()
print("Train dataset size:", len(train_dataset.dataset))
print("Test dataset size:", len(test_dataset.dataset))


# ============================================================
# 2. YAPAY SİNİR AĞI MODELİ
# ============================================================

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
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# ============================================================
# 3. EĞİTİM VE TEST
# ============================================================

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    epoch_losses = []

    for epoch in range(num_epochs):
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            prediction = model(images)
            loss = criterion(prediction, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Loss grafiğini kaydet ve göster
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('training_loss.png')
    plt.show()

MODEL_PATH = "ann_model.pth"

if os.path.exists(MODEL_PATH):
    print(f"\n[BİLGİ] Kayıtlı model ağırlıkları bulundu: {MODEL_PATH}, yükleniyor...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
else:
    print("\n[BİLGİ] Model eğitiliyor (10 epoch)...")
    train_model(model, train_dataset, criterion, optimizer, num_epochs=10)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[BİLGİ] Model kaydedildi: {MODEL_PATH}")

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

test_model(model, test_dataset)


# ============================================================
# 4. KENDİ EL YAZISI RESMİNDEN 10 RAKAMI BULUP TAHMİN ETME
# ============================================================

def segment_and_predict(model, digits_folder="./my_digits"):
    """
    my_digits klasöründeki resmi okur,
    0-9 rakamlarını tespit edip MNIST formatına çevirir ve tahmin eder.
    """
    model.eval()

    image_files = sorted([
        f for f in os.listdir(digits_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        and not f.lower().startswith('sonuc')
        and not f.lower().startswith('tahmin_')
    ])

    if not image_files:
        print(f"{digits_folder} klasöründe resim bulunamadı!")
        return

    fpath = os.path.join(digits_folder, image_files[0])
    print(f"\nResim okunuyor: {fpath}")

    img_bgr = cv2.imread(fpath)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h_img, w_img = img_gray.shape

    # Arka plan siyah mı beyaz mı?
    if img_gray.mean() < 127:
        # Siyah arka plan, beyaz rakamlar (Paint gibi)
        _, thresh = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY)
    else:
        # Beyaz kağıt, koyu rakamlar
        img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 51, 15
        )

    # Konturları bul
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Rakamların çizgilerini kalınlaştır (MNIST standardı kalın çizgilere benzetir)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 500 and w > 10 and h > 20:  # gürültü filtresi
            boxes.append((x, y, w, h))

    if not boxes:
        print("Hiç rakam tespit edilemedi! Resmi kontrol edin.")
        return

    # Satırlara göre sırala (Örn: üst satır 0-4, alt satır 5-9)
    mid_y = h_img / 2.0
    row1 = sorted([b for b in boxes if b[1] < mid_y], key=lambda b: b[0])
    row2 = sorted([b for b in boxes if b[1] >= mid_y], key=lambda b: b[0])

    if len(row1) > 0 and len(row2) > 0:
        sorted_boxes = row1 + row2
    else:
        sorted_boxes = sorted(boxes, key=lambda b: b[0])

    print(f"Toplam {len(sorted_boxes)} adet rakam tespit edildi.\n")

    digit_imgs = []
    predictions = []
    confidences = []

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    for (x, y, w, h) in sorted_boxes:
        roi = dilated[y:y+h, x:x+w]

        # En-boy oranını koruyarak kare yap
        pad = abs(h - w) // 2
        if h > w:
            roi_square = cv2.copyMakeBorder(roi, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
        else:
            roi_square = cv2.copyMakeBorder(roi, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)

        # 20x20 resize + 4 piksel border -> 28x28 (MNIST standardı)
        roi_20 = cv2.resize(roi_square, (20, 20), interpolation=cv2.INTER_AREA)
        roi_28 = cv2.copyMakeBorder(roi_20, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)

        roi_pil = Image.fromarray(roi_28).convert('L')
        tensor = transform(roi_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)

        predictions.append(pred.item())
        confidences.append(conf.item() * 100)
        digit_imgs.append(roi_28)

    # Sonuçları yazdır
    print("=" * 45)
    print(f"{'Sıra':<6} {'Beklenen':^10} {'Tahmin':^10} {'Güven':^10}")
    print("=" * 45)
    dogru_sayisi = 0
    for i, (pred, conf) in enumerate(zip(predictions, confidences)):
        beklenen = str(i) if i < 10 else "?"
        is_correct = "[DOGRU]" if beklenen == str(pred) else "[YANLIS]"
        if beklenen == str(pred):
            dogru_sayisi += 1
        print(f"{i+1:<6} {beklenen:^10} {str(pred):^10} %{conf:^8.1f} {is_correct}")
    print("=" * 45)
    print(f"Toplam Basari: {dogru_sayisi}/{len(predictions)} (%{100*dogru_sayisi/len(predictions):.0f})\n")

    # Görselleştirme: Grid
    n = len(digit_imgs)
    cols = min(5, n)
    rows = (n + cols - 1) // cols

    plt.figure(figsize=(3 * cols, 3 * rows + 1))
    for i, (img, pred, conf) in enumerate(zip(digit_imgs, predictions, confidences)):
        beklenen = str(i) if i < 10 else "?"
        color = 'green' if beklenen == str(pred) else 'red'
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"B: {beklenen} | T: {pred}\n%{conf:.0f}", fontsize=11, color=color)
        plt.axis('off')
    plt.suptitle("El Yazısı Rakam Tahminleri", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(digits_folder, "tahmin_grid.png"))
    plt.show()

    # Orijinal resim üzerine kutuları ve tahminleri çiz
    img_result = img_bgr.copy()
    for i, (x, y, w, h) in enumerate(sorted_boxes):
        beklenen = str(i) if i < 10 else "?"
        color = (0, 255, 0) if beklenen == str(predictions[i]) else (0, 0, 255)
        cv2.rectangle(img_result, (x, y), (x + w, y + h), color, 3)
        cv2.putText(
            img_result,
            f"{predictions[i]}",
            (x + 5, y + 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            color,
            3
        )

    result_path = os.path.join(digits_folder, "sonuc.jpg")
    cv2.imwrite(result_path, img_result)
    print(f"Tahminli sonuç resmi kaydedildi: {result_path}")

    result_rgb = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 6))
    plt.imshow(result_rgb)
    plt.title("Rakamlar ve Model Tahminleri (Yeşil: Doğru, Kırmızı: Yanlış)", fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(digits_folder, "sonuc_goruntusu.png"))
    plt.show()

# Tahmini çalıştır
segment_and_predict(model, digits_folder="./my_digits")
