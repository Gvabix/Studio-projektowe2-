# Studio-projektowe2
[RST github link](https://github.com/jiyfeng/DPLP)
[Link do artykułu](https://arxiv.org/abs/1903.09196)
# Pierwsze kroki
Skoro funkcja **RST** jest już zaimplementowana i pobrana z GitHuba, możemy skupić się na wstępnych krokach organizacji projektu oraz jego strukturze.  

---

## ** Początkowe kroki**  

### **1️⃣ Utworzenie repozytorium i środowiska pracy**  
- [ ] Stwórz repozytorium na **GitHub/GitLab** dla całego zespołu.  
- [ ] Skonfiguruj wirtualne środowisko **(Python venv/conda)** do pracy nad projektem.  
- [ ] Zainstaluj wymagane biblioteki (**numpy, pandas, sklearn, networkx, nltk, torch, transformers itp.**).  
- [ ]  Pobierz i umieść kod **RST** w repozytorium, sprawdzając jego działanie na przykładowych danych.  

```bash
git clone <repo_url>
cd project_name
python -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)
pip install -r requirements.txt
```

---

### **2️⃣ Struktura katalogów i plików**  

```
/fake_news_detection_project
│── data/                  # Folder na dane wejściowe i przetworzone
│   ├── raw/               # Surowe dane (FakeNewsNet)
│   ├── processed/         # Wstępnie przetworzone dane
│
│── models/                # Folder na modele ML/NLP
│   ├── rst_model.py       # Kod RST z GitHuba
│   ├── hpnf_model.py      # Implementacja HPNF
│   ├── combined_model.py  # Połączenie RST + HPNF
│
│── notebooks/             # Eksploracja danych i testy modeli
│   ├── data_analysis.ipynb
│   ├── model_evaluation.ipynb
│
│── scripts/               # Skrypty do obróbki i analizy danych
│   ├── preprocess_data.py # Czyszczenie i przygotowanie danych
│   ├── train_model.py     # Trenowanie modeli
│   ├── evaluate_model.py  # Ewaluacja wyników
│
│── reports/               # Raporty i wyniki
│   ├── progress_report.md
│   ├── final_report.md
│
│── main.py                # Główna logika aplikacji
│── requirements.txt        # Lista wymaganych bibliotek
│── README.md               # Opis projektu i instrukcje
```

---

### **3️⃣ Przygotowanie i analiza danych**  
- [ ] Pobierz zbiór **FakeNewsNet** i umieść go w folderze `/data/raw/`.  
- [ ] Stwórz skrypt `preprocess_data.py`, który oczyści i przygotuje dane do analizy.  
- [ ] Sprawdź, czy format danych jest zgodny z wejściem wymaganym przez RST.
- [ ] Przetestuj działanie RST na próbce danych.  

---

### **4️⃣ Implementacja metody HPNF**  
- [ ] Stwórz plik `hpnf_model.py` i zaimplementuj analizę propagacji fake newsów.  
- [ ] Użyj biblioteki **networkx** do tworzenia sieci retweetów i komentarzy.
- [ ] Wygeneruj wskaźniki propagacji (np. głębokość drzewa, czas reakcji użytkowników).

#### Implementacja HPNF

##### **1. Wstępne przygotowanie danych**
   - Zbierz i przygotuj dane wejściowe do modelu.
   - Przekształć dane do postaci odpowiedniej dla modelu (np. normalizacja, tokenizacja, transformacja do tensorów).
   - Podziel dane na zbiór treningowy, walidacyjny i testowy.

##### **2. Definicja architektury HPNF**
   - **Wejście**: Punktowe wartości danych w formie sekwencji.
   - **Hierarchiczne warstwy**:
     - Warstwy filtrujące pierwszego rzędu przekształcające surowe dane na bardziej reprezentatywne cechy.
     - Warstwy agregujące łączące informacje z poprzednich poziomów.
   - **Mechanizm uwagi** (attention mechanism) do przetwarzania zależności w danych sekwencyjnych.
   - **Moduł wyjściowy**: Finalna warstwa w pełni połączona (fully connected) lub warstwa konwolucyjna, w zależności od typu zadania.

##### **3. Trenowanie modelu**
   - Wybór funkcji straty (np. błąd średniokwadratowy MSE dla regresji).
   - Wybór optymalizatora (np. Adam, SGD).
   - Wykorzystanie mini-batch gradient descent do aktualizacji wag.
   - Regularizacja (dropout, L2-norm) dla poprawy generalizacji.

##### **4. Ewaluacja i optymalizacja**
   - Przeprowadzenie testów na zbiorze walidacyjnym.
   - Modyfikacja hiperparametrów (np. liczba warstw, wielkość batcha, stopa uczenia).
   - Analiza wyników modelu przy użyciu metryk (np. RMSE, MAE, F1-score).

##### **5. Wdrożenie modelu**
   - Optymalizacja obliczeniowa dla wydajności (pruning, quantization).
   - Eksport modelu w wybranym formacie (np. ONNX, TensorFlow SavedModel).
   - Integracja z systemem produkcyjnym.
---

### **5️⃣ Integracja RST + HPNF**  
- [ ] Stwórz plik `combined_model.py`, łącząc wyniki z RST i HPNF.
- [ ] Przetestuj, czy połączenie metod zwiększa skuteczność detekcji fake newsów.
- [ ] Przygotuj eksperymenty porównujące skuteczność RST, HPNF i ich połączenia.  
