import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from collections import Counter
from gensim import corpora, models
import re

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords

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
        <h1 class='judul'>Hotel Review Sentiment Analyzer</h1>
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
        <h1 class='judul'>Dashboard Visualisasi Sentimen</h1>
    """, unsafe_allow_html=True)

    if st.button("Kembali ke Upload", key="back-btn"):
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

    st.subheader("Tabel Data Review")
    tabel_tampil = df[['review', 'channel', 'sentimen', 'rating']].dropna().head(1000)
    st.dataframe(tabel_tampil, height=300, use_container_width=True)

    with st.expander("Lihat semua review utuh"):
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

            wedges, texts, autotexts = ax1.pie(sentimen_counts, 
                                            labels=sentimen_counts.index, 
                                            autopct='%1.1f%%', 
                                            startangle=90, 
                                            colors=['#4caf50', '#ff9800', '#f44336'],
                                            wedgeprops={'edgecolor': 'black', 'linewidth': 1, 'linestyle': 'solid'},
                                            pctdistance=0.8,
                                            labeldistance=1.1) 
            
            for text in texts:
                text.set_fontsize(8)
                text.set_horizontalalignment('center')

            for autotext in autotexts:
                autotext.set_fontsize(7)
                autotext.set_color('black') 

            ax1.axis('equal') 

            ax1.legend(wedges, sentimen_counts.index, title="Sentimen", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

            st.pyplot(fig1)

    with col2:
        with st.expander("📊 Rata-Rata Rating per Kategori Sentimen"):
            st.markdown("<div class='description'>Distribusi rating menurut sentimen untuk melihat pola penilaian ulasan.</div>", unsafe_allow_html=True)

            # Menyiapkan visualisasi
            fig2, ax2 = plt.subplots(figsize=(6, 4))

            # Grupkan data berdasarkan sentimen dan hitung rata-rata rating
            avg_rating_per_sentiment = df.groupby('sentimen')['rating'].mean()

            # Plot rata-rata rating berdasarkan sentimen
            avg_rating_per_sentiment.plot(kind='bar', ax=ax2, color=['#4caf50', '#f44336', '#ff9800'], alpha=0.7)

            # Label dan title
            ax2.set_xlabel("Sentimen")
            ax2.set_ylabel("Rata-Rata Rating")
            ax2.set_title("Rata-Rata Rating Berdasarkan Sentimen")
            ax2.set_xticklabels(avg_rating_per_sentiment.index, rotation=0)

            # Menambahkan penjelasan pada grafik
            st.pyplot(fig2)

   # ------------------------ N-GRAM ANALYSIS ------------------------
    st.subheader("Analisis Frekuensi Kata")
    with st.expander("📊 Kata-Kata yang Sering Muncul Bersamaan dalam Ulasan"):
        def get_top_ngrams(texts, n=2, top_k=10):
            all_ngrams = []
            for text in texts:
                tokens = [w.lower() for w in nltk.word_tokenize(text) if w.isalpha() and w.lower() not in stop_words]
                n_grams = ngrams(tokens, n)
                all_ngrams.extend(n_grams)
            counter = Counter(all_ngrams)
            return counter.most_common(top_k)

        bigrams = get_top_ngrams(df['review'].dropna().astype(str), n=2, top_k=10)
        bigram_df = pd.DataFrame(bigrams, columns=["Bigram", "Frequency"])
        bigram_df["Bigram"] = bigram_df["Bigram"].apply(lambda x: ' '.join(x))

        st.markdown("""
        <div style='text-align: justify'>
        Bigram menunjukkan pasangan kata yang sering muncul bersama dalam ulasan. 
        Ini membantu melihat hal-hal yang paling sering disebut tamu, baik positif maupun negatif.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Tabel Top 10 Bigram**")
            st.dataframe(bigram_df, use_container_width=True)

        with col2:
            st.markdown("**Visualisasi Frekuensi Bigram**")
            st.bar_chart(bigram_df.set_index("Bigram"))

    # ------------------------ TOPIC MODELING (LDA) ------------------------
    st.subheader("Topik Utama Analisis Review")
    with st.expander("📊 Hal yang Sering Dibicarakan"):
        st.markdown(
            "Pengelompokan isi review ke dalam beberapa topik utama. "
            "Setiap topik mewakili kumpulan kata yang sering muncul bersama dan menggambarkan tema tertentu dari ulasan pelanggan."
        )

        # Preprocessing
        def clean_text(text):
            text = re.sub(r'[^a-zA-Z]', ' ', text)
            return text.lower()

        texts = df['review'].dropna().astype(str).apply(clean_text).tolist()
        tokenized_texts = [
            [word for word in nltk.word_tokenize(t) if word not in stop_words and len(word) > 2]
            for t in texts
        ]

        # Dictionary dan Corpus
        dictionary = corpora.Dictionary(tokenized_texts)
        corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

        # LDA Model
        lda_model = models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=10)

        # Tampilkan topik dengan kata kunci
        for i, topic in lda_model.print_topics(num_words=5):
            # Ambil kata-kata kunci dan pisahkan
            keywords = [word.split('*')[1].strip().strip('"') for word in topic.split(' + ')]

            # Tampilkan topik dengan kata kunci tanpa label
            st.markdown(f"**Topik {i+1}:**")
            st.markdown("`" + "`, `".join(keywords) + "`")
