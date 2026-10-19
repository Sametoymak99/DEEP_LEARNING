
import torch #matris işlemleri için
import torch.nn as nn  #yapay sinir ağı katmanlarını tanımlamak için kullanılır
import torch.optim as optim #optimizasyon algoritmalarını kontrol eden bölüm
import torchvision #görüntü işleme önceden eğitilmiş modelleri içerir
import torchvision.transforms as transforms # görüntü dönüşümleri yapmak için kullanılır
import matplotlib.pyplot as plt 

device = torch.device("cuda" if torch.cuda.is_available() 
                      else "cpu")

#veri seti yükleme ve ön işleme kısmı

def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    #mnist veri setini indir ve test ve train kümelerini oluşturur
    train_set = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    #pytorch veri yükleyicisini oluşturur
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


# data visualization

def visualization_samples(data_loader):
    data_iter = iter(data_loader)
    images, labels = next(data_iter)

    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f"Label: {labels[i].item()}")
        plt.axis('off')
    plt.show()


visualization_samples(train_dataset)


#yapay sinir ağı classı

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


#model oluştur ve derleme kısmı

model = NeuralNetwork().to(device)

#kayıp fonksiyonu ve optimizasyon algoritması tanımlanır

define_loss_and_optimizer = lambda model: (
    nn.CrossEntropyLoss(),
    optim.Adam(model.parameters(), lr=0.001)
)


#train

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    epoch_losses = []  # Her epoch'un kaybını saklar

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

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Loss: {avg_loss:.4f}"
        )

    #loss grafiği

    plt.figure(figsize=(10, 5))
    plt.plot(
        range(1, num_epochs + 1),
        epoch_losses,
        marker='o'
    )
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()


criterion, optimizer = define_loss_and_optimizer(model)
train_model(model, train_dataset, criterion, optimizer, num_epochs=10)


#test

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
