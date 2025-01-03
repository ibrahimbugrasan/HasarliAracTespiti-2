#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 20:27:16 2025

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
from sklearn.preprocessing import LabelBinarizer
from transformers import DeiTForImageClassification
from google.colab import drive

# DeiT Base Distilled modeli yükleyelim
model_name = 'facebook/deit-base-distilled-patch16-224'
model = DeiTForImageClassification.from_pretrained(model_name, num_labels=4, ignore_mismatched_sizes=True)
model.classifier = torch.nn.Linear(model.classifier.in_features, 4)

# Google Drive üzerinden veri yükleme kısmı 
drive.mount('/content/drive')

# Veriyi burada uygun şekilde yükleyin (örneğin, 'all_data' klasörü altında)
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

# Yardımcı Fonksiyonlar
def plot_roc_curve(y_true, y_score, class_names, epoch):
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

def plot_confusion_matrix(cm, class_names, epoch):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix for Epoch {epoch+1}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def plot_loss(train_losses, epoch):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label='Train Loss')
    plt.title('Epoch Loss Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(range(1, len(train_losses) + 1))
    plt.legend()
    plt.show()

# Performans Metrik Hesaplama
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)

    # Sensitivity (Duyarlılık) ve Specificity (Özgüllük) hesaplama
    sensitivity = recall_score(y_true, y_pred, average=None)  # her sınıf için duyarlılık
    specificity = []
    for i in range(len(cm)):
        tn = cm[i].sum() - cm[i][i]  # True Negative
        fp = cm[:, i].sum() - cm[i][i]  # False Positive
        specificity.append(tn / (tn + fp) if tn + fp != 0 else 0)

    # AUC hesaplama
    lb = LabelBinarizer()
    y_true_bin = lb.fit_transform(y_true)
    auc_value = roc_auc_score(y_true_bin, y_pred, multi_class='ovr')

    return accuracy, recall, precision, f1, sensitivity, specificity, auc_value, cm

# Eğitim Fonksiyonu
def train_model(model, train_loader, test_loader, epochs=10):
    train_losses = []
    class_names = full_data.classes  # Sınıf isimlerini buradan alıyoruz

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

        train_losses.append(running_loss / len(train_loader))

        # Test (Doğrulama)
        model.eval()
        all_preds = []
        all_labels = []
        all_outputs = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).logits
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_outputs.extend(outputs.cpu().numpy())

        # One-hot encoding için LabelBinarizer kullanarak all_labels'ı dönüştür
        lb = LabelBinarizer()
        all_labels_one_hot = lb.fit_transform(all_labels)

        # Softmax ile olasılıkları elde et
        all_outputs_softmax = torch.softmax(torch.tensor(all_outputs), dim=1).numpy()

        # Performans metriklerini hesapla
        accuracy, recall, precision, f1, sensitivity, specificity, auc_value, cm = calculate_metrics(all_labels, all_preds)

        # Grafikleri çiz
        plot_roc_curve(all_labels_one_hot, all_outputs_softmax, class_names, epoch)
        plot_confusion_matrix(cm, class_names, epoch)
        plot_loss(train_losses, epoch)

        # Metrikleri yazdır
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_losses[-1]:.4f}")
        print(f"Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1-Score: {f1:.4f}")
        print(f"Sensitivity: {sensitivity}, Specificity: {specificity}")
        print(f"AUC: {auc_value:.4f}")

# Modeli Eğitme
train_model(model, train_loader, test_loader, epochs=10)