__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import pandas as pd
import datetime
import re
import os
from langchain_groq import ChatGroq

# --- SAYFA YAPISI ---
st.set_page_config(page_title="Okul Akıllı Asistanı", page_icon="🏫", layout="wide")

# --- GELİŞMİŞ GÖRSEL TASARIM (CSS) ---
st.markdown("""
    <style>
    .main { background: linear-gradient(135deg, #0e1117 0%, #161b22 100%); }
    .main-title {
        background: linear-gradient(90deg, #00dbde 0%, #fc00ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 3.5rem !important;
        font-weight: 800;
        margin-bottom: 10px;
    }
    .card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease, border 0.3s ease;
        height: 320px; /* Yeni sorular için biraz uzatıldı */
    }
    .card:hover {
        transform: translateY(-10px);
        border: 1px solid rgba(0, 219, 222, 0.5);
    }
    .card-1 { border-left: 8px solid #00dbde; }
    .card-2 { border-left: 8px solid #fc00ff; }
    .card h3 { color: #ffffff; font-size: 1.5rem; margin-bottom: 15px; }
    .card ul { padding-left: 20px; color: #d1d5db; font-size: 0.9rem; }
    .card li { margin-bottom: 8px; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>🏫 Okul Arkadaşım AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#888;'>Yapay Zeka Destekli Ders ve Mevzuat Rehberi</p>", unsafe_allow_html=True)

# --- SECRETS & LLM ---
if "GROQ_API_KEY" not in st.secrets:
    st.error("GROQ_API_KEY bulunamadı!")
    st.stop()

llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0, api_key=st.secrets["GROQ_API_KEY"])

# --- VERİ YÜKLEME ---
@st.cache_data
def load_data():
    try:
        return pd.read_csv('SinifProgrami1404.csv', sep=';')
    except:
        return None

df = load_data()

# --- VİTRİN KARTLARI (YENİ SORULAR EKLENDİ) ---
col1, col2 = st.columns(2)

with col1:
    st.markdown("""<div class="card card-1"><h3>📅 Ders Programı Örnekleri</h3>
    <ul>
        <li>• 11-A sınıfı Çarşamba 4. saatte dersi nerede işleyecek?</li>
        <li>• 11-B sınıfı Cuma günü hangi dersi var?</li>
        <li>• 9-E sınıfı Salı 2. saatte hangi öğretmenin dersi var?</li>
    </ul></div>""", unsafe_allow_html=True)

with col2:
    st.markdown("""<div class="card card-2"><h3>⚖️ Mevzuat Rehberi Örnekleri</h3>
    <ul>
        <li>• 8 gün devamsızlık belge almama engel mi?</li>
        <li>• Yıl sonu ortalamam 52, sınıfı geçebilir miyim?</li>
        <li>• Sorumluluk sınavından kaç alırsam geçerim?</li>
    </ul></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- ASİSTAN FONKSİYONU (COLAB MANTIĞI) ---
def asistan_sorgula(soru):
    gunler_tr = {0: "Pazartesi", 1: "Salı", 2: "Çarşamba", 3: "Perşembe", 4: "Cuma", 5: "Cumartesi", 6: "Pazar"}
    bugun_adi = gunler_tr[datetime.datetime.now().weekday()]
    yarin_adi = gunler_tr[(datetime.datetime.now().weekday() + 1) % 7]

    nlu_prompt = f"Soru: '{soru}' Bugün: {bugun_adi}, Yarın: {yarin_adi}. Bilgileri SINIF:[...], GUN:[...], SAAT:[...] formatında çek."

    try:
        cikti = llm.invoke(nlu_prompt).content
        h_sinif = re.search(r"SINIF:\[?(.*?)\]?,", cikti).group(1).replace("-", "").strip()
        h_gun = re.search(r"GUN:\[?(.*?)\]?,", cikti).group(1).strip()
        h_saat = re.search(r"SAAT:\[?(.*?)\]?$", cikti).group(1).strip()

        mask = (df['Sinif'].str.replace("-", "").str.contains(h_sinif, case=False, na=False)) & \
               (df['Gun'].str.contains(h_gun, case=False, na=False))
        if h_saat.isdigit(): mask = mask & (df['Girilen Ders Saati'] == int(h_saat))
        
        sonuc = df[mask].to_string(index=False) if not df[mask].empty else "Kayıt bulunamadı."
        
        final_msg = f"Tablo verisine göre öğrenciye kısa ve net cevap ver. VERİ: {sonuc}\nSORU: {soru}"
        return llm.invoke(final_msg).content
    except:
        return "Üzgünüm, sorunu tam anlayamadım. Lütfen sınıf ve zaman belirterek tekrar sorabilir misin? 😊"

# --- SOHBET ALANI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Neyi merak ediyorsun?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        response = asistan_sorgula(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
