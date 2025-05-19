import zipfile
import os

# Ścieżka do pliku ZIP
zip_path = os.path.join('..', 'data', 'graphs.zip')

# Ścieżka do folderu docelowego
extract_to = os.path.join('..', 'data')

os.makedirs(extract_to, exist_ok=True)

# Rozpakowywanie
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print(f'Plik {zip_path} został rozpakowany do {extract_to}')
