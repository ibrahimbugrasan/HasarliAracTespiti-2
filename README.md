# **HasarliAracTespiti**  
**Hasarlı Araç Tahmini ve Görüntü İşleme Projesi**

## **Proje Özeti**  
Bu proje, hasarlı araç görüntüleri üzerinde derin öğrenme tabanlı transformatör modelleri kullanarak sınıflandırma ve tahmin işlemleri yapmayı hedeflemektedir. Toplanan veriler üzerinde **Beit**, **Swin**, **Deit**, **VGG16**, ve **ConvNext** modelleri uygulanmış; her modelin performansı çeşitli metriklerle analiz edilmiştir. Sonuçlar, raporlanarak modellerin başarımı karşılaştırılmıştır.

## **Proje Özellikleri**  
- **Farklı transformatör modelleri** ile sınıflandırma.  
- Performans metrikleri:  
  - **Accuracy**, **Recall**, **Precision**, **Sensitivity**, **Specificity**, **F-Score**, ve **AUC** hesaplamaları.  
- **Karmaşıklık matrisi** ve **ROC eğrisi** analizi.  
- Eğitim/test veri kümeleri için **epoch vs. loss** grafikleri.  
- **Eğitim zamanı (training time)** ve **çıkarım zamanı (inference time)** analizi.  
- IEEE konferans şablonuna uygun detaylı raporlama.

## **Geliştirme Ortamı**  
Proje geliştirme ve model eğitimi şu ortamda gerçekleştirilmiştir:  

- **Programlama Dili:** Python 3.8+  
- **Derin Öğrenme Frameworkleri:** PyTorch, TensorFlow  
- **Kütüphaneler:**  
  - `numpy`, `pandas`, `matplotlib`, `seaborn`  
  - `scikit-learn`, `torchvision`, `tqdm`  
  - `transformers`, `opencv-python`  
- **Donanım:**  
  - NVIDIA GPU (örnek: Tesla T4, 16 GB VRAM)  
