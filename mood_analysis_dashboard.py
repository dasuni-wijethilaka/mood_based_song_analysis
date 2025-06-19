import streamlit as st
import pandas as pd
import re

st.set_page_config(page_title="🎵 Mood Analysis Dashboard", layout="wide")

st.title("🎵 Song Mood Analysis Dashboard")
st.markdown("Upload your **CSV** file to analyze song moods and emotions.")

# Keyword matching
def contains_keywords(text, keywords):
    return bool(re.search(rf'\b({keywords})\b', str(text), re.IGNORECASE))

# Mood analysis logic
def analyze_moods(df):
    df = df.dropna(subset=['song', 'key', 'energy', 'tempo', 'Lyrics'])
    df['key'] = pd.to_numeric(df['key'], errors='coerce')
    df['energy'] = pd.to_numeric(df['energy'], errors='coerce')
    df['tempo'] = pd.to_numeric(df['tempo'], errors='coerce')

    results = []
    for _, row in df.iterrows():
        key = row['key']
        energy = row['energy']
        tempo = row['tempo']
        lyrics = row['Lyrics']

        if pd.isna(key) or pd.isna(energy) or pd.isna(tempo):
            continue

        mood_scores = {
            'sad': sum([
                key < 5,
                energy < 0.4,
                tempo < 90,
                contains_keywords(lyrics, "sorry|cry|lonely|sad|goodbye|fall")
            ]) / 4,
            'happy': sum([
                energy > 0.6,
                tempo > 110,
                key in [2, 4, 7, 11],
                contains_keywords(lyrics, "fun|smile|party|sunshine|joy")
            ]) / 4,
            'energetic': sum([
                energy > 0.7,
                tempo > 120,
                key in [2, 4, 7, 11],
                contains_keywords(lyrics, "fire|dance|wild|burn|crazy|alive")
            ]) / 4,
            'romantic': sum([
                0.4 <= energy <= 0.7,
                80 <= tempo <= 115,
                key in [0, 7, 9],
                contains_keywords(lyrics, "love|heart|kiss|baby|darling|hold")
            ]) / 4,
            'relaxing': sum([
                energy < 0.5,
                70 <= tempo <= 110,
                key in [5, 7],
                contains_keywords(lyrics, "calm|peace|slow|breeze|chill|dream")
            ]) / 4,
        }

        sorted_moods = sorted(mood_scores.items(), key=lambda x: x[1], reverse=True)
        major = sorted_moods[0][0]
        second = sorted_moods[1][0]

        results.append({
            'song': row['song'],
            **{f'{m}_weight': w for m, w in mood_scores.items()},
            'major_feeling': major,
            'second_major_feeling': second
        })

    return pd.DataFrame(results)

# File upload UI
uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        results_df = analyze_moods(df)
        results_df['song_normalized'] = results_df['song'].str.strip().str.lower()

        # Try loading the ground truth file
        try:
            ground_truth = pd.read_csv("song_mood_predictions.csv")
            ground_truth['song'] = ground_truth['song'].str.strip().str.lower()
            ground_truth['major_feeling'] = ground_truth['major_feeling'].str.strip().str.lower()

            merged_df = results_df.merge(
                ground_truth[['song', 'major_feeling', 'second_major_feeling']],
                left_on='song_normalized', right_on='song', how='left',
                suffixes=('', '_actual')
            )

            merged_df['is_correct'] = merged_df['major_feeling'] == merged_df['major_feeling_actual']
            accuracy = merged_df['is_correct'].mean()
        except Exception as e:
            st.warning("⚠️ Ground truth file not found or invalid. Skipping accuracy comparison.")
            merged_df = results_df
            accuracy = None

        st.success("✅ Analysis completed!")

        # Stats
        st.subheader("📊 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Songs", len(results_df))
        col2.metric("Avg Happy Score", f"{results_df['happy_weight'].mean()*100:.0f}%")
        col3.metric("Avg Sad Score", f"{results_df['sad_weight'].mean()*100:.0f}%")
        top_mood = results_df['major_feeling'].mode().iloc[0]
        col4.metric("Most Common Mood", top_mood.capitalize())

        # Accuracy
        if accuracy is not None:
            st.metric("🎯 Prediction Accuracy", f"{accuracy:.2%}")

        # Detailed Table
        st.subheader("🎼 Detailed Mood Analysis")
        st.dataframe(results_df[['song', 'sad_weight', 'happy_weight', 'energetic_weight', 'romantic_weight', 'relaxing_weight', 'major_feeling', 'second_major_feeling']])

        # Mood-wise Tabs
        st.subheader("🎯 Songs by Mood Categories")
        tabs = st.tabs(["😊 Happy", "😢 Sad", "💕 Romantic", "🧘 Relaxing", "⚡ Energetic"])
        moods = ['happy', 'sad', 'romantic', 'relaxing', 'energetic']

        for i, mood in enumerate(moods):
            with tabs[i]:
                filtered = results_df[results_df['major_feeling'] == mood]
                st.write(f"**{len(filtered)} songs classified as {mood.capitalize()}**")
                st.dataframe(filtered[['song', f'{mood}_weight', 'second_major_feeling']]
                             .rename(columns={f'{mood}_weight': 'weight'})
                             .sort_values(by='weight', ascending=False))

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

else:
    st.info("Upload a CSV file to begin analysis.")
