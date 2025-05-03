import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

# ------------------------ SETUP ------------------------
st.set_page_config(page_title="Hotel Review Sentiment", layout="wide")
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ------------------------ STATE INIT ------------------------
if 'page' not in st.session_state:
    st.session_state['page'] = 'upload'

# ------------------------ HALAMAN UPLOAD DATA ------------------------
if st.session_state['page'] == 'upload':
    st.markdown("""
        <h1 class='judul'>🛎️ Hotel Review Sentiment Analyzer</h1>
        <p class='subtext'>Analisis ulasan pelanggan hotel dengan visualisasi menarik</p>
        <hr class='custom'>
    """, unsafe_allow_html=True)

    st.header("📤 Upload Dataset")
    uploaded_file = st.file_uploader("Upload file CSV (wajib ada kolom 'review' ya)", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'review' not in df.columns:
            st.error("❌ Upsie kolom 'review' tidak ditemukan nich")
        else:
            st.session_state['data'] = df
            st.success("✅ yeay data berhasil diupload dan disimpan")
            st.dataframe(df.head(1000))

            if st.button("Lihat Visualisasi", key="next-btn"):
                st.session_state['page'] = 'dashboard'
                st.rerun()

# ------------------------ HALAMAN DASHBOARD ------------------------
elif st.session_state['page'] == 'dashboard':
    st.markdown("""
        <h1 class='judul'>📊 Dashboard Visualisasi Sentimen</h1>
    """, unsafe_allow_html=True)

    if st.button("⬅️ Kembali ke Upload", key="back-btn"):
        st.session_state['page'] = 'upload'
        st.rerun()

    df = st.session_state['data']

    def get_sentiment(text):
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0.1:
            return 'Positif'
        elif polarity < -0.1:
            return 'Negatif'
        else:
            return 'Netral'

    df['sentimen'] = df['review'].astype(str).apply(get_sentiment)

    def normalize_rating(r):
        try:
            r = float(r)
            if r > 5:
                return round(r / 2, 1)
            return round(r, 1)
        except:
            return None

    if 'rating' in df.columns:
        df['rating'] = df['rating'].apply(normalize_rating)
    else:
        df['rating'] = None

    if 'channel' not in df.columns:
        df['channel'] = 'Lainnya'

    st.subheader("📋 Tabel Data Review")
    tabel_tampil = df[['review', 'channel', 'sentimen', 'rating']].dropna().head(1000)
    st.dataframe(tabel_tampil, height=300, use_container_width=True)

    with st.expander("📋 Lihat semua review utuh"):
        for idx, row in tabel_tampil.iterrows():
            st.markdown(
                f"<div class='review-box'><b>{row['sentimen']}</b> – <i>{row['channel']}</i>, ⭐ {row['rating']}<br>{row['review']}</div>",
                unsafe_allow_html=True
            )

    st.subheader("📈 Ringkasan Sentimen")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Review", len(df))
    col2.metric("Positif", (df['sentimen'] == 'Positif').sum())
    col3.metric("Negatif", (df['sentimen'] == 'Negatif').sum())
    col4.metric("Netral", (df['sentimen'] == 'Netral').sum())

    st.subheader("Visualisasi Review Sentimen")
    st.markdown("<div class='description'>Wordcloud berikut menampilkan kata-kata yang paling sering muncul. " \
    "Semakin besar ukuran kata, semakin sering kata tersebut disebutkan.</div>", unsafe_allow_html=True)
    col_wc1, col_wc2 = st.columns(2)
    for sentimen, col in zip(['Positif', 'Negatif'], [col_wc1, col_wc2]):
        with col:
            st.markdown(f"**{sentimen}**")
            text = ' '.join(df[df['sentimen'] == sentimen]['review'].dropna().astype(str))
            if text:
                wordcloud = WordCloud(width=500, height=300, background_color='white').generate(text)
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
            else:
                st.info("Tidak ada data untuk ditampilkan.")


    col1, col2 = st.columns(2) 
    
    with col1:
        with st.expander("📊 Proporsi Sentimen"):
            st.markdown("<div class='description'>Chart ini menunjukkan distribusi ulasan berdasarkan sentimen: positif, netral, dan negatif. " \
            "Berguna untuk melihat sebaran persepsi pelanggan secara umum.</div>", unsafe_allow_html=True)
            fig1, ax1 = plt.subplots(figsize=(5, 3))
            sentimen_counts = df['sentimen'].value_counts()
            ax1.pie(sentimen_counts, labels=sentimen_counts.index, autopct='%1.1f%%', startangle=90, colors=['#4caf50', '#ff9800', '#f44336'])
            ax1.axis('equal')
            st.pyplot(fig1)

    with col2:
        with st.expander("📊 Distribusi Rating"):
            st.markdown("<div class='description'>Visualisasi ini memperlihatkan rata-rata rating yang diberikan pengguna untuk tiap sentimen. " \
            "Membantu memahami apakah ulasan positif selalu disertai rating tinggi, dan sebaliknya.</div>", unsafe_allow_html=True)
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            rating_counts = df['rating'].value_counts().sort_index()
            ax2.bar(rating_counts.index.astype(str), rating_counts.values, color='#60a5fa')
            ax2.set_xlabel("Rating")
            ax2.set_ylabel("Jumlah Review")
            ax2.set_title("Distribusi Rating Review")
            st.pyplot(fig2)