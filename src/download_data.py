#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Veri indirme işlemlerini otomatize etmek için yardımcı script.
Colab ve diğer ortamlarda data/ klasörü yoksa otomatik olarak oluşturur
ve ilgili veri dosyalarını indirip kaydeder.
"""

import os
import pandas as pd
import requests
import io
import sys
from urllib.parse import urlparse
from pathlib import Path

# Google Drive'dan paylaşılan dosyaları indirmek için
def download_file_from_google_drive(id, destination):
    """
    Google Drive'dan paylaşılan bir dosyayı indirir.
    
    Parametreler:
    - id: Google Drive dosya ID'si
    - destination: Dosyanın kaydedileceği yer
    """
    URL = "https://drive.google.com/uc?export=download"
    
    session = requests.Session()
    
    response = session.get(URL, params={'id': id, 'confirm': 't'}, stream=True)
    token = get_confirm_token(response)
    
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    
    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

# Kaggle'dan veri indirme (API'ye alternatif yöntem)
def download_from_url(url, destination):
    """
    Verilen URL'den dosyayı indirir ve belirtilen hedefe kaydeder.
    
    Parametreler:
    - url: İndirilecek dosyanın URL'si
    - destination: Dosyanın kaydedileceği yer
    """
    r = requests.get(url)
    r.raise_for_status()  # Başarısız ise hata yükselt
    
    with open(destination, 'wb') as f:
        f.write(r.content)
    
    print(f"Dosya {destination} konumuna indirildi.")

def download_from_github(repo_url, file_path, destination):
    """
    GitHub'dan bir dosyayı ham (raw) biçimde indirir.
    
    Parametreler:
    - repo_url: GitHub repo URL'si (örn. 'https://github.com/username/repo')
    - file_path: Repo içindeki dosya yolu (örn. 'data/file.csv')
    - destination: Dosyanın kaydedileceği yer
    """
    # GitHub repo URL'sini raw URL'ye dönüştür
    parsed_url = urlparse(repo_url)
    path_parts = parsed_url.path.strip('/').split('/')
    if len(path_parts) < 2:
        raise ValueError("Geçersiz GitHub repo URL'si")
    
    username, repo = path_parts[:2]
    raw_url = f"https://raw.githubusercontent.com/{username}/{repo}/main/{file_path}"
    
    try:
        r = requests.get(raw_url)
        r.raise_for_status()
        
        with open(destination, 'wb') as f:
            f.write(r.content)
        
        print(f"Dosya {destination} konumuna indirildi.")
    except requests.exceptions.RequestException as e:
        print(f"Dosya indirme hatası: {e}")
        # Ana dal 'main' yerine 'master' olabilir, tekrar dene
        raw_url = f"https://raw.githubusercontent.com/{username}/{repo}/master/{file_path}"
        r = requests.get(raw_url)
        r.raise_for_status()
        
        with open(destination, 'wb') as f:
            f.write(r.content)
        
        print(f"Dosya {destination} konumuna indirildi.")

def check_and_download_data(data_dir='data'):
    """
    Veri dizinini kontrol eder ve eksik dosyaları indirir.
    
    Parametreler:
    - data_dir: Veri dizini, varsayılan olarak 'data'
    """
    # Veri dizinini oluştur
    os.makedirs(data_dir, exist_ok=True)
      # İndirilecek dosyaların listesi (URL'ler ve hedef dosya adları)
    files_to_download = {
        # Örnek: Google Drive linki
        'skyserver.csv': {
            'type': 'gdrive',
            'id': '1DeOt59I6usxGFh2Tm4eBzZi3148p5CUj',  # Gerçek Drive ID'si
            'backup_url': 'https://drive.google.com/uc?export=download&id=1DeOt59I6usxGFh2Tm4eBzZi3148p5CUj'
        },
        'star_subtypes.csv': {
            'type': 'gdrive',
            'id': '1H6DWrepHH36ErofbHRv0BzB8L4EaGb2i',  # Gerçek Drive ID'si
            'backup_url': 'https://drive.google.com/uc?export=download&id=1H6DWrepHH36ErofbHRv0BzB8L4EaGb2i'
        }
        # Daha fazla dosya eklenebilir
    }
    
    # Her dosyayı kontrol et ve yoksa indir
    for filename, file_info in files_to_download.items():
        file_path = os.path.join(data_dir, filename)
        
        # Dosya var mı kontrol et
        if os.path.exists(file_path):
            print(f"{filename} dosyası zaten mevcut.")
            continue  
        
        print(f"{filename} indiriliyor...")
        
        try:
            if file_info['type'] == 'gdrive':
                try:
                    download_file_from_google_drive(file_info['id'], file_path)
                except Exception as e:
                    print(f"Google Drive'dan indirme başarısız: {e}")
                    print("Yedek URL'den indirmeyi deneniyor...")
                    download_from_url(file_info['backup_url'], file_path)
            elif file_info['type'] == 'url':
                download_from_url(file_info['url'], file_path)
            elif file_info['type'] == 'github':
                download_from_github(
                    file_info['repo_url'], 
                    file_info['file_path'], 
                    file_path
                )
        except Exception as e:
            print(f"Dosya indirme hatası: {e}")
            print("Devam etmek için dosyaları manuel olarak yüklemeniz gerekebilir.")
            continue
        
        print(f"{filename} başarıyla indirildi.")

def check_missing_data():
    """
    Eksik veri dosyalarını kontrol eder ve kullanıcıya bilgi verir.
    
    Returns:
    - bool: Tüm veri dosyaları mevcut ise True, eksik varsa False
    """
    data_dir = 'data'
    required_files = ['skyserver.csv', 'star_subtypes.csv']
    
    if not os.path.exists(data_dir):
        print(f"'{data_dir}' dizini bulunamadı.")
        return False
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        print("Aşağıdaki veri dosyaları eksik:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    return True

def get_download_instruction():
    """
    Veri dosyalarını indirmek için kullanım talimatlarını döndürür.
    """
    instruction = """
    # Veri Dosyalarını İndirme Talimatları
    
    Gerekli veri dosyalarını indirmek için aşağıdaki kodu çalıştırın:
    
    ```python
    from download_data import check_and_download_data
    check_and_download_data()
    ```
    
    Veya doğrudan terminal/komut satırından:
    
    ```
    python download_data.py
    ```
    
    İndirme başarısız olursa, dosyaları manuel olarak indirip 'data/' dizinine yerleştirin.
    """
    return instruction

if __name__ == "__main__":
    print("Veri dosyaları kontrol ediliyor ve indiriliyor...")
    check_and_download_data()
    print("\nVeri indirme işlemi tamamlandı. Artık modelinizi eğitebilirsiniz.")
