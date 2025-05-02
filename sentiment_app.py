# sentiment_app.py
import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud

st.set_page_config(page_title="Hotel Review Sentiment App", layout="centered")

st.title("📊 Hotel Review Sentiment Analyzer")
st.write("Yuks upload data ulasan hotel kamu untuk melihat analisis sentimennya.")

uploaded_file = st.file_uploader("Upload file CSV (dengan kolom 'review')", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    if 'review' not in df.columns:
        st.error("ih dataset kamu gak ada kolom 'review' nya deh")
    else:
        st.success("yeay filenya berhasil dimuat!")
        
        # Fungsi sentiment analysis
        def get_sentiment(text):
            polarity = TextBlob(text).sentiment.polarity
            if polarity > 0.1:
                return 'Positif'
            elif polarity < -0.1:
                return 'Negatif'
            else:
                return 'Netral'

        df['sentimen'] = df['review'].astype(str).apply(get_sentiment)

        # Pie chart sentimen
        st.subheader("📈 Distribusi Sentimen")
        sentimen_count = df['sentimen'].value_counts()
        fig1, ax1 = plt.subplots()
        ax1.pie(sentimen_count, labels=sentimen_count.index, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)

        # Wordcloud
        st.subheader("☁️ Wordcloud Ulasan")
        all_text = ' '.join(df['review'].dropna().astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.imshow(wordcloud, interpolation='bilinear')
        ax2.axis("off")
        st.pyplot(fig2)

        # Tampilkan contoh review
        st.subheader("📝 Contoh Review")
        for kategori in ['Positif', 'Netral', 'Negatif']:
            st.markdown(f"**{kategori}**")
            st.write(df[df['sentimen'] == kategori]['review'].head(2).to_list())
