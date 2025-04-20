import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

def generate_wordclouds():
    # Load datasets again (only for visualization)
    df_fake = pd.read_csv('data/Fake.csv')
    df_real = pd.read_csv('data/True.csv')

    # Combine title + text
    df_fake['content'] = df_fake['title'] + " " + df_fake['text']
    df_real['content'] = df_real['title'] + " " + df_real['text']

    # Generate text for each
    fake_text = " ".join(df_fake['content'].astype(str).tolist())
    real_text = " ".join(df_real['content'].astype(str).tolist())

    # Make sure output folder exists
    os.makedirs("outputs", exist_ok=True)

    # Fake News WordCloud
    fake_wc = WordCloud(width=800, height=400, background_color='black').generate(fake_text)
    plt.figure(figsize=(12, 6))
    plt.imshow(fake_wc, interpolation='bilinear')
    plt.axis('off')
    plt.title("Fake News WordCloud")
    plt.savefig("outputs/wordcloud_fake.png")
    plt.close()

    # Real News WordCloud
    real_wc = WordCloud(width=800, height=400, background_color='white').generate(real_text)
    plt.figure(figsize=(12, 6))
    plt.imshow(real_wc, interpolation='bilinear')
    plt.axis('off')
    plt.title("Real News WordCloud")
    plt.savefig("outputs/wordcloud_real.png")
    plt.close()

    print("✅ WordClouds generated and saved to outputs/")
