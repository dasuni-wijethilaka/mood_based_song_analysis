import streamlit as st
import pandas as pd
import re
import os

st.set_page_config(page_title="🎵 Song Mood Analysis Dashboard", layout="wide")

st.title("🎵 Song Mood Analysis Dashboard")
st.markdown("Upload your **CSV** file to analyze song moods and emotions.")

os.makedirs("dataset", exist_ok=True)

def contains_keywords(text, keywords):
    return bool(re.search(rf'\b({keywords})\b', str(text), re.IGNORECASE))

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
        singer = row.get('singer', 'Unknown')

        if pd.isna(key) or pd.isna(energy) or pd.isna(tempo):
            continue

        mood_scores = {
            'sad': sum([
                key in [2, 3] and energy < 0.4,
                energy < 0.35,
                tempo < 85,
                contains_keywords(lyrics, "sorry|cry|lonely|sad|goodbye|fall|apart|miss|nothing|lasts")
            ]) / 4,

            'happy': sum([
                energy > 0.75,
                tempo > 120,
                key in [4, 7, 9],
                contains_keywords(lyrics, "party|smile|sunshine|fun|up")
            ]) / 4,

            'energetic': sum([
                energy > 0.8,
                tempo > 125,
                key in [4],
                contains_keywords(lyrics, "alive|fire|wild|rush|brave|shake")
            ]) / 4,

            'romantic': sum([
                0.45 <= energy <= 0.65,
                80 <= tempo <= 100,
                key in [0, 2, 9],
                contains_keywords(lyrics, "love|heart|kiss|baby|darling|hold")
            ]) / 4,

            'relaxing': sum([
                energy < 0.4,
                70 <= tempo <= 90,
                key in [2, 3, 7],
                contains_keywords(lyrics, "fall|good|calm|peace|slow|heal") and not contains_keywords(lyrics, "cry|sad|sorry")
            ]) / 4,
        }

        weighted_scores = {}
        for mood, score in mood_scores.items():
            if mood == 'sad':
                key_score = int(key in [2, 3] and energy < 0.4)
                energy_score = int(energy < 0.35)
                tempo_score = int(tempo < 85)
                lyric_score = int(contains_keywords(lyrics, "sorry|cry|lonely|sad|goodbye|fall|apart|miss|nothing"))
                weighted_scores[mood] = (energy_score * 0.3 + lyric_score * 0.4 + tempo_score * 0.2 + key_score * 0.1)

            elif mood == 'happy':
                key_score = int(key in [4, 7, 9])
                energy_score = int(energy > 0.75)
                tempo_score = int(tempo > 120)
                lyric_score = int(contains_keywords(lyrics, "party|smile|sunshine|fun|make me smile"))
                weighted_scores[mood] = (energy_score * 0.3 + lyric_score * 0.4 + tempo_score * 0.2 + key_score * 0.1)

            elif mood == 'energetic':
                key_score = int(key in [4])
                energy_score = int(energy > 0.8)
                tempo_score = int(tempo > 125)
                lyric_score = int(contains_keywords(lyrics, "alive|fire|wild|rush|shake"))
                weighted_scores[mood] = (energy_score * 0.35 + lyric_score * 0.35 + tempo_score * 0.25 + key_score * 0.05)

            elif mood == 'romantic':
                key_score = int(key in [0, 2, 9])
                energy_score = int(0.45 <= energy <= 0.65)
                tempo_score = int(80 <= tempo <= 100)
                lyric_score = int(contains_keywords(lyrics, "love|heart|kiss|baby|darling|hold|you|mine"))
                weighted_scores[mood] = (energy_score * 0.2 + lyric_score * 0.5 + tempo_score * 0.2 + key_score * 0.1)

            elif mood == 'relaxing':
                key_score = int(key in [2, 3, 7])
                energy_score = int(energy < 0.4)
                tempo_score = int(70 <= tempo <= 90)
                lyric_score = int(contains_keywords(lyrics, "sky|fall|good|calm|peace|slow") and not contains_keywords(lyrics, "cry|sad|sorry"))
                weighted_scores[mood] = (energy_score * 0.3 + lyric_score * 0.3 + tempo_score * 0.3 + key_score * 0.1)

        sorted_moods = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
        major = sorted_moods[0][0]
        second = sorted_moods[1][0]

        results.append({
            'song': row['song'],
            'singer': singer,
            'tempo': tempo,
            'energy': energy,
            **{f'{m}_weight': w for m, w in weighted_scores.items()},
            'major_feeling': major,
            'second_major_feeling': second
        })

    return pd.DataFrame(results)

uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        results_df = analyze_moods(df)
        results_df['song_normalized'] = results_df['song'].str.strip().str.lower()

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

            st.subheader("🔍 Prediction vs Actual Comparison")
            comparison_df = merged_df[['song', 'major_feeling', 'major_feeling_actual', 'is_correct']].copy()
            comparison_df.columns = ['Song', 'Predicted', 'Actual', 'Correct']
            st.dataframe(comparison_df)

        except Exception:
            st.warning("⚠️ Ground truth file not found or invalid. Skipping accuracy comparison.")
            merged_df = results_df
            accuracy = None

        st.success("✅ Analysis completed!")

        st.subheader("📊 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Songs", len(results_df))
        col2.metric("Avg Happy Score", f"{results_df['happy_weight'].mean()*100:.0f}%")
        col3.metric("Avg Sad Score", f"{results_df['sad_weight'].mean()*100:.0f}%")
        top_mood = results_df['major_feeling'].mode().iloc[0]
        col4.metric("Most Common Mood", top_mood.capitalize())

        if accuracy is not None:
            st.metric("🎯 Prediction Accuracy", f"{accuracy:.2%}")

        st.subheader("🎼 Detailed Mood Analysis")
        display_df = results_df[['song', 'sad_weight', 'happy_weight', 'energetic_weight', 'romantic_weight', 'relaxing_weight', 'major_feeling', 'second_major_feeling']].copy()
        for col in ['sad_weight', 'happy_weight', 'energetic_weight', 'romantic_weight', 'relaxing_weight']:
            display_df[col] = display_df[col].round(3)
        st.dataframe(display_df)

        st.subheader("🎯 Songs by Mood Categories")
        unique_moods = sorted(results_df['major_feeling'].unique())
        mood_emojis = {
            'happy': '😊',
            'sad': '😢',
            'romantic': '💕',
            'relaxing': '🧘',
            'energetic': '⚡'
        }

        tab_labels = [f"{mood_emojis.get(mood, '🎵')} {mood.capitalize()}" for mood in unique_moods]
        tabs = st.tabs(tab_labels)

        for i, mood in enumerate(unique_moods):
            with tabs[i]:
                filtered = results_df[results_df['major_feeling'] == mood].copy()
                st.write(f"**{len(filtered)} songs classified as {mood.capitalize()}**")
                if len(filtered) > 0:
                    display_columns = ['song', f'{mood}_weight', 'second_major_feeling']
                    display_filtered = filtered[display_columns].copy()
                    display_filtered = display_filtered.rename(columns={f'{mood}_weight': f'{mood.capitalize()} Score'})
                    display_filtered[f'{mood.capitalize()} Score'] = display_filtered[f'{mood.capitalize()} Score'].round(3)
                    display_filtered = display_filtered.rename(columns={'second_major_feeling': 'Secondary Mood'})
                    display_filtered = display_filtered.sort_values(by=f'{mood.capitalize()} Score', ascending=False)
                    st.dataframe(display_filtered, use_container_width=True)
                    avg_score = filtered[f'{mood}_weight'].mean()
                    max_score = filtered[f'{mood}_weight'].max()
                    st.write(f"Average {mood.capitalize()} Score: **{avg_score:.3f}** | Highest Score: **{max_score:.3f}**")
                else:
                    st.write("No songs found for this mood category.")

        # st.subheader("📥 Download Personalized Playlist")
        csv_columns = ['song', 'singer', 'tempo', 'energy', 'sad_weight', 'happy_weight',
                      'energetic_weight', 'romantic_weight', 'relaxing_weight',
                      'major_feeling', 'second_major_feeling']
        personalized_playlist = results_df[csv_columns].copy()
        weight_columns = ['sad_weight', 'happy_weight', 'energetic_weight', 'romantic_weight', 'relaxing_weight']
        for col in weight_columns:
            personalized_playlist[col] = personalized_playlist[col].round(3)
        personalized_playlist['tempo'] = personalized_playlist['tempo'].round(1)
        personalized_playlist['energy'] = personalized_playlist['energy'].round(3)

        # ✅ Save CSV in current directory
        personalized_playlist.to_csv("dataset/personalized_playlist.csv", index=False)

        # Prepare for download
        # csv_data = personalized_playlist.to_csv(index=False)

        # st.download_button(
        #     label="📥 Download Personalized Playlist CSV",
        #     data=csv_data,
        #     file_name="personalized_playlist.csv",
        #     mime="text/csv",
        #     help="Download the complete mood analysis results as a CSV file"
        # )

        st.write("**Preview of Personalized Playlist CSV:**")
        st.dataframe(personalized_playlist.head(10), use_container_width=True)
        if len(personalized_playlist) > 10:
            st.write(f"*Showing first 10 rows out of {len(personalized_playlist)} total songs*")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
else:
    st.info("Upload a CSV file to begin analysis.")
