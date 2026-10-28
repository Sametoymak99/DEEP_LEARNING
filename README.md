# MNIST El Yazısı Rakam Tanıma Ödevi (Yapay Sinir Ağı - ANN)

Bu projede PyTorch kullanılarak MNIST veri seti üzerinde bir Yapay Sinir Ağı (ANN) eğitilmiş ve el yazısıyla çizilmiş 0-9 arası rakamların OpenCV ile tespiti ve sınıflandırılması yapılmıştır.

## 📌 Proje İçeriği
- nn.py: Model mimarisi, eğitim/test ve el yazısı rakam tanıma kodları.
- nn_model.pth: Eğitilmiş ANN model ağırlıkları.
- my_digits/: El yazısıyla çizilmiş test görseli (1.png) ve modelin tahmin sonuçları (sonuc.jpg).

## 🚀 Çalıştırma

Gerekli kütüphaneleri yükleyin:
`ash
pip install torch torchvision opencv-python pillow matplotlib numpy
`

Projeyi çalıştırmak için:
`ash
python ann.py
`
