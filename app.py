import streamlit as st
import pandas as pd
import datetime
import re
from langchain_groq import ChatGroq

# --- SAYFA YAPISI ---
st.set_page_config(page_title="Ders Programı Asistanı", page_icon="📅", layout="wide")

# --- GÖRSEL TASARIM (CSS) ---
st.markdown("""
    <style>
    .main { background: linear-gradient(135deg, #0e1117 0%, #161b22 100%); }
    .stTitle { 
        background: linear-gradient(90deg, #0083ff 0%, #00d4ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center; 
        font-size: 3rem !important; 
        font-weight: 800;
        margin-bottom: 25px; 
    }
    .example-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border-left: 5px solid #0083ff;
        margin-bottom: 20px;
    }
    .example-card h4 { color: #00d4ff; margin-bottom: 10px; }
    .example-card ul { color: #d1d5db; list-style-type: none; padding-left: 0; }
    .example-card li { margin-bottom: 8px; font-size: 0.95rem; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 class='stTitle'>📅 Akıllı Ders Programı Asistanı</h1>", unsafe_allow_html=True)

# --- ÖRNEK SORULAR PANALİ ---
st.markdown("""
    <div class="example-card">
        <h4>💡 Örnek Sorular</h4>
        <ul>
            <li>• 11-A sınıfı Çarşamba 4. saatte dersi nerede işleyecek?</li>
            <li>• 11-B sınıfı Cuma günü hangi dersi var?</li>
            <li>• 9-E sınıfı Salı 2. saatte hangi öğretmenin dersi var?</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# --- SECRETS & LLM YAPILANDIRMASI ---
if "GROQ_API_KEY" not in st.secrets:
    st.error("Lütfen Streamlit Secrets paneline GROQ_API_KEY ekleyin.")
    st.stop()

llm = ChatGroq(
    model_name="llama-3.1-8b-instant", 
    temperature=0, 
    api_key=st.secrets["GROQ_API_KEY"]
)

# --- VERİ YÜKLEME ---
@st.cache_data
def load_data():
    try:
        return pd.read_csv('SinifProgrami1404.csv', sep=';')
    except FileNotFoundError:
        st.error("Hata: 'SinifProgrami1404.csv' dosyası bulunamadı!")
        return None

df = load_data()

# --- ASİSTAN MANTIĞI (COLAB İLE %100 UYUMLU) ---
def ogrenci_asistani_kesin_cozum(soru):
    gunler_tr = {0: "Pazartesi", 1: "Salı", 2: "Çarşamba", 3: "Perşembe", 4: "Cuma", 5: "Cumartesi", 6: "Pazar"}
    bugun_adi = gunler_tr[datetime.datetime.now().weekday()]
    yarin_adi = gunler_tr[(datetime.datetime.now().weekday() + 1) % 7]

    nlu_prompt = f"""
    Soru: "{soru}"
    Bugün: {bugun_adi}, Yarın: {yarin_adi}
    Bu sorudaki SINIF, GUN ve SAAT bilgilerini sadece şu formatta yaz: SINIF:[...], GUN:[...], SAAT:[...]
    """

    try:
        cikti = llm.invoke(nlu_prompt).content
        
        # Sizin Colab'da kullandığınız Regex yapısı (Hatasız Çekim)
        h_sinif = re.search(r"SINIF:\[?(.*?)\]?,", cikti).group(1).replace("-", "").strip()
        h_gun = re.search(r"GUN:\[?(.*?)\]?,", cikti).group(1).strip()
        h_saat = re.search(r"SAAT:\[?(.*?)\]?$", cikti).group(1).strip()

        # Pandas Filtreleme
        mask = (df['Sinif'].str.replace("-", "").str.contains(h_sinif, case=False, na=False)) & \
               (df['Gun'].str.contains(h_gun, case=False, na=False))

        if h_saat.isdigit():
            mask = mask & (df['Girilen Ders Saati'] == int(h_saat))

        sonuc_df = df[mask]
        context = sonuc_df.to_string(index=False) if not sonuc_df.empty else "Kayıt bulunamadı."
        
    except Exception:
        return "Üzgünüm, sınıfını veya hangi günü sorduğunu tam anlayamadım. (Örn: 10-A yarın 2. saat ne var?)"

    final_prompt = f"""
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

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ders programını sor (Örn: 9A bugün 3. saat ne?)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Program kontrol ediliyor..."):
                response = ogrenci_asistani_kesin_cozum(prompt)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
