import streamlit as st
import pandas as pd
import datetime
import re
from langchain_groq import ChatGroq

# --- SAYFA YAPISI ---
st.set_page_config(page_title="Okul Ders Programı", page_icon="📅", layout="wide")

# --- GÖRSEL TASARIM (CSS) ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stTitle { color: white; text-align: center; font-size: 2.5rem !important; margin-bottom: 20px; }
    .card {
        background-color: #1a1c24;
        border-radius: 15px;
        padding: 15px;
        border-top: 5px solid #0083ff;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 class='stTitle'>📅 Akıllı Ders Programı Asistanı</h1>", unsafe_allow_html=True)

# --- SECRETS & LLM YAPILANDIRMASI ---
if "GROQ_API_KEY" not in st.secrets:
    st.error("Lütfen Streamlit Secrets paneline GROQ_API_KEY ekleyin.")
    st.stop()

# Colab'daki sıcaklık ve model ayarlarıyla birebir aynı
llm = ChatGroq(
    model_name="llama-3.1-8b-instant", 
    temperature=0, 
    api_key=st.secrets["GROQ_API_KEY"]
)

# --- VERİ YÜKLEME ---
@st.cache_data
def load_data():
    # Colab'daki CSV okuma mantığı: Ayırıcı ';'
    try:
        # Colab'daki CSV okuma mantığı: Ayırıcı ';'
        return pd.read_csv('SinifProgrami1404.csv', sep=';')
    except FileNotFoundError:
        st.error("Hata: 'SinifProgrami1404.csv' dosyası bulunamadı!")
        return None

df = load_data()

# --- ASİSTAN MANTIĞI (COLAB İLE BİREBİR) ---
# --- ASİSTAN MANTIĞI (COLAB İLE %100 UYUMLU) ---
def ogrenci_asistani_kesin_cozum(soru):
    # Gün hesaplama mantığı
    gunler_tr = {0: "Pazartesi", 1: "Salı", 2: "Çarşamba", 3: "Perşembe", 4: "Cuma", 5: "Cumartesi", 6: "Pazar"}
    bugun_adi = gunler_tr[datetime.datetime.now().weekday()]
    yarin_adi = gunler_tr[(datetime.datetime.now().weekday() + 1) % 7]

    # NLU: Sorudan veri ayıklama
    nlu_prompt = f"""
    Soru: "{soru}"
    Bugün: {bugun_adi}, Yarın: {yarin_adi}
    Bu sorudaki SINIF, GUN ve SAAT bilgilerini sadece şu formatta yaz: SINIF:[...], GUN:[...], SAAT:[...]
    """

    try:
        cikti = llm.invoke(nlu_prompt).content

        # Düzenli ifadelerle veriyi ayıklama (Colab mantığı)
        h_sinif = re.search(r"SINIF:\[?(.*?)\],", cikti).group(1).replace("-", "").strip()
        h_gun = re.search(r"GUN:\[?(.*?)\],", cikti).group(1).strip()
        h_saat = re.search(r"SAAT:\[?(.*?)$", cikti).group(1).replace("]", "").strip()
        # COLAB'DAKİ %100 ÇALIŞAN REGEX DESENLERİ (GÜNCELLENDİ)
        # Sınıf, gün ve saati parantez olsa da olmasa da yakalar
        h_sinif = re.search(r"SINIF:\[?(.*?)\]?,", cikti).group(1).replace("-", "").strip()
        h_gun = re.search(r"GUN:\[?(.*?)\]?,", cikti).group(1).strip()
        h_saat = re.search(r"SAAT:\[?(.*?)\]?$", cikti).group(1).strip()

        # Pandas Filtreleme (Maske)
        # Sınıf ismindeki "-" işaretlerini kaldırarak eşleştirme yapar (10-A ve 10A aynı sayılır)
        mask = (df['Sinif'].str.replace("-", "").str.contains(h_sinif, case=False, na=False)) & \
               (df['Gun'].str.contains(h_gun, case=False, na=False))

        if h_saat.isdigit():
            mask = mask & (df['Girilen Ders Saati'] == int(h_saat))

        sonuc_df = df[mask]

        # Sonuç bağlamı oluşturma
        context = sonuc_df.to_string(index=False) if not sonuc_df.empty else "Kayıt bulunamadı."

    except Exception:
        return "Üzgünüm, sınıfını veya hangi günü sorduğunu tam anlayamadım. (Örn: 10A yarın 2. saat ne var?)"
        return "Üzgünüm, sınıfını veya hangi günü sorduğunu tam anlayamadım. (Örn: 10A Pazartesi 3. saat ne var?)"

    # Final Yanıt Oluşturma (Colab mantığı)
    # Final Yanıt Oluşturma
    final_prompt = f"""
    Sen bir okul asistanısın. Aşağıdaki tablo verisini kullanarak öğrenciye net ve kısa bir cevap ver.
    Sen bir okul asistanısın. Aşağıdaki tablo verisini kullanarak öğrenciye net, samimi ve kısa bir cevap ver.
    Veride olmayan bilgiyi uydurma.
    
    TABLO VERİSİ:
    {context}
    
    ÖĞRENCİ SORUSU: 
    {soru}
    """
    return llm.invoke(final_prompt).content

# --- SOHBET AKIŞI ---
if df is not None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Eski mesajları ekrana bas
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Yeni soru girişi
    if prompt := st.chat_input("Ders programını sor (Örn: 9A bugün 3. saat ne?)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Program kontrol ediliyor..."):
                response = ogrenci_asistani_kesin_cozum(prompt)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
