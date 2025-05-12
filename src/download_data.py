#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Veri dosyalarının varlığını kontrol eden basit bir script.
Eksik dosyalar varsa kullanıcıyı uyarır.
"""

import os
import sys

def check_data_files(data_dir='data'):
    """
    Veri dosyalarının varlığını kontrol eder ve eksikse uyarı verir.
    
    Parametreler:
    - data_dir: Veri dizini, varsayılan olarak 'data'
    
    Returns:
    - bool: Tüm veri dosyaları mevcut ise True, eksik varsa False
    """
    required_files = ['skyserver.csv', 'star_subtypes.csv']
    
    if not os.path.exists(data_dir):
        print(f"'{data_dir}' dizini bulunamadı.")
        print(f"Lütfen '{data_dir}' klasörünü oluşturun.")
        return False
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        print("\n⚠️  Aşağıdaki veri dosyaları eksik:")
        for file in missing_files:
            print(f"  - {file}")
        
        print("\nLütfen eksik dosyaları manuel olarak 'data/' klasörüne yükleyin.")
        print("Colab'da dosyaları yüklemek için sol paneldeki dosya simgesini kullanabilirsiniz.")
        print("1. Sol paneldeki dosya simgesine tıklayın")
        print("2. 'data/' klasörüne tıklayın (yoksa oluşturun)")
        print("3. Yükle butonunu kullanarak dosyaları yükleyin")
        return False
    
    print("Tüm veri dosyaları mevcut. Modeli eğitmeye devam edebilirsiniz.")
    return True

if __name__ == "__main__":
    print("Veri dosyaları kontrol ediliyor...")
    if check_data_files():
        print("\nKontrol başarılı, tüm dosyalar mevcut.")
    else:
        print("\nBazı dosyalar eksik, lütfen bunları manuel olarak yükleyin.")
