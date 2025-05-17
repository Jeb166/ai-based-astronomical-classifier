"""
Astronomik Sınıflandırıcı Uygulaması - Ana Başlatıcı

Bu script, Streamlit uygulamasını başlatır.
Çalıştırmak için: streamlit run main.py
"""

import subprocess
import os
import sys

def run_streamlit_app():
    """Streamlit uygulamasını başlatır"""
    # Mevcut dosyanın bulunduğu dizini belirle
    current_dir = os.path.dirname(os.path.abspath(__file__))
    streamlit_file = os.path.join(current_dir, 'src', 'streamlit.py')
    
    # Streamlit uygulamasını başlat
    subprocess.run([sys.executable, "-m", "streamlit", "run", streamlit_file])

if __name__ == "__main__":
    run_streamlit_app()
