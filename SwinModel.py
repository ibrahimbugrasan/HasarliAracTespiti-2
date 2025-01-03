#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 20:21:53 2025

@author: ibrahimbugrasan
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from transformers import SwinForImageClassification, SwinConfig
from google.colab import drive

# Swin modelini yükleyelim
model_name = 'microsoft/swin-tiny-patch4-window7-224'
model = SwinForImageClassification.from_pretrained(model_name, num_labels=4, ignore_mismatched_sizes=True)
model.classifier = torch.nn.Linear(model.classifier.in_features, 4)

# Google Drive üzerinden veri yükleme kısmı
drive.mount('/content/drive')

# Veriyi ölçeklendirme
full_data = torchvision.datasets.ImageFolder('/content/drive/MyDrive/YazLabProjesi/DATABASE', transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]))

# Eğitim ve test verisi ayırma işlemi 
train_size = int(0.8 * len(full_data))
test_size = len(full_data) - train_size
train_data, test_data = torch.utils.data.random_split(full_data, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False)

# Eğitim Ayarları
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Eğitim Fonksiyonu
def train_model(model, train_loader, test_loader, epochs=10,):
    train_losses, test_losses = [], []
    accuracies, recalls, precisions, f1_scores, aucs = [], [], [], [], []
    confusion_matrices = []
    epoch_times, inference_times = [], []
    epoch_train_losses = []  # Train loss for each epoch
    epoch_test_losses = []   # Test loss for each epoch

    print("Model eğitimine başlıyor...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        # Eğitim
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        epoch_train_losses.append(running_loss / len(train_loader))

        # Test 
        model.eval()
        all_preds = []
        all_labels = []
        all_outputs = []  # Sınıf olasılıklarını tutmak için ekliyoruz
        start_inference = time.time()

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).logits
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_outputs.extend(outputs.cpu().numpy())  # Olasılıkları kaydediyoruz

        # One-hot encoding için LabelBinarizer kullanarak all_labels'ı dönüştür
        lb = LabelBinarizer()
        all_labels_one_hot = lb.fit_transform(all_labels)

        # Softmax ile olasılıkları elde et
        all_outputs_softmax = torch.softmax(torch.tensor(all_outputs), dim=1).numpy()

        # ROC eğrisini hesapla ve çiz
        plot_roc_curve(all_labels_one_hot, all_outputs_softmax, lb.classes_, epoch)

        # Accuracy, Recall, Precision, F1, AUC hesaplama
        accuracy = accuracy_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds, average='weighted')
        precision = precision_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        auc_value = roc_auc_score(all_labels, all_outputs_softmax, multi_class='ovr', average='weighted')

        accuracies.append(accuracy)
        recalls.append(recall)
        precisions.append(precision)
        f1_scores.append(f1)
        aucs.append(auc_value)

        # Confusion matrix hesaplama
        cm = confusion_matrix(all_labels, all_preds)
        confusion_matrices.append(cm)

        # Sensitivity (Duyarlılık) ve Specificity (Özgüllük) hesaplama
        sensitivity = []
        specificity = []
        for i in range(4):  # 4 sınıf için
            tn, fp, fn, tp = cm[i, i], cm[i, :].sum() - cm[i, i], cm[:, i].sum() - cm[i, i], cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
            sensitivity.append(tp / (tp + fn) if (tp + fn) > 0 else 0)  # True Positive Rate
            specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0)  # True Negative Rate

        # Epoch sürelerini ve inference zamanını yazdırma
        inference_time = time.time() - start_inference
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {running_loss / len(train_loader):.4f}")
        print(f"Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")
        print(f"Sensitivity: {sensitivity}, Specificity: {specificity}")
        print(f"F1 Score: {f1:.4f}, AUC: {auc_value:.4f}")
        print(f"Epoch Time: {epoch_time:.2f}s, Inference Time: {inference_time:.2f}s")

        epoch_test_losses.append(criterion(torch.tensor(all_outputs), torch.tensor(all_labels)).item())  # Test loss

        # Grafiklerin gösterimi
        plot_loss_graph(epoch_train_losses, epoch_test_losses)
        plot_confusion_matrix(cm, epoch)

# Yardımcı Fonksiyonlar
def plot_loss_graph(train_losses, test_losses):
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Train Loss', color='blue', linestyle='-', marker='o')
    plt.plot(test_losses, label='Test Loss', color='red', linestyle='-', marker='x')
    plt.legend()
    plt.title('Training and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(cm, epoch):
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pert', 'Çizik', 'Göçük', 'Camkırığı'], yticklabels=['Pert', 'Çizik', 'Göçük', 'Camkırığı'])
    plt.title(f'Epoch {epoch+1} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def plot_roc_curve(y_true, y_score, class_names, epoch):
    # ROC Eğrisini Çizme
    plt.figure(figsize=(10, 8))
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
        auc_value = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {auc_value:.2f})')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Epoch {epoch+1}')
    plt.legend(loc='lower right')
    plt.show()

# Modeli Eğitme
train_model(model, train_loader, test_loader, epochs=10,)