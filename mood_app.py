import streamlit as st
import pandas as pd
import re

def generate_feeling_tables(file_path):
    df = pd.read_csv(file_path)

    # Clean data
    df = df[['song', 'key', 'energy', 'tempo', 'Lyrics'] + ([col for col in df.columns if col == 'actual_mood'])].dropna()
    df['key'] = pd.to_numeric(df['key'], errors='coerce')
    df['energy'] = pd.to_numeric(df['energy'], errors='coerce')
    df['tempo'] = pd.to_numeric(df['tempo'], errors='coerce')

    # Mood conditions
    moods = {
        'happy': [
            lambda r: r['energy'] > 0.6,
            lambda r: r['tempo'] > 110,
            lambda r: r['key'] in [2, 4, 7, 11],
            lambda r: bool(re.search(r'\b(fun|smile|party|sunshine|joy)\b', str(r['Lyrics']), re.IGNORECASE))
        ],
        'sad': [
            lambda r: r['key'] < 5,
            lambda r: r['energy'] < 0.4,
            lambda r: r['tempo'] < 90,
            lambda r: bool(re.search(r'\b(sorry|cry|lonely|sad|goodbye|fall)\b', str(r['Lyrics']), re.IGNORECASE))
        ],
        'romantic': [
            lambda r: 0.4 <= r['energy'] <= 0.7,
            lambda r: 80 <= r['tempo'] <= 115,
            lambda r: r['key'] in [0, 7, 9],
            lambda r: bool(re.search(r'\b(love|heart|kiss|baby|darling|hold)\b', str(r['Lyrics']), re.IGNORECASE))
        ],
        'relaxing': [
            lambda r: r['energy'] < 0.5,
            lambda r: 70 <= r['tempo'] <= 110,
            lambda r: r['key'] in [5, 7],
            lambda r: bool(re.search(r'\b(calm|peace|slow|breeze|chill|dream)\b', str(r['Lyrics']), re.IGNORECASE))
        ],
        'energetic': [
            lambda r: r['energy'] > 0.7,
            lambda r: r['tempo'] < 120,
            lambda r: r['key'] in [2, 4, 7, 11],
            lambda r: bool(re.search(r'\b(fire|dance|wild|burn|crazy|alive)\b', str(r['Lyrics']), re.IGNORECASE))
        ]
    }

    # Calculate mood weights
    for mood, checks in moods.items():
        df[f'{mood}_weight'] = df.apply(lambda r: sum(check(r) for check in checks) / len(checks), axis=1)

    # Determine major feeling and its weight
    weight_cols = [f"{m}_weight" for m in moods]
    df['major_feeling'] = df[weight_cols].idxmax(axis=1).str.replace('_weight', '')
    df['major_weight'] = df[weight_cols].max(axis=1)
    
        # Accuracy calculation if actual labels are present
    if 'actual_mood' in df.columns:
        df['actual_mood'] = df['actual_mood'].str.lower().str.strip()
        df['is_correct'] = df['actual_mood'] == df['major_feeling']
        accuracy = df['is_correct'].mean()  # This gives a float like 0.85
    else:
        accuracy = None
    mood_tables, accuracy = generate_feeling_tables(uploaded_file)
    if accuracy is not None:
        st.success(f"🎯 Mood Prediction Accuracy: {accuracy:.2%}")
    else:
        st.warning("⚠️ 'actual_mood' column not found in uploaded CSV. Accuracy can't be calculated.")


    # Calculate accuracy if ground truth exists
    if 'actual_mood' in df.columns:
        df['actual_mood'] = df['actual_mood'].str.lower().str.strip()
        df['is_correct'] = df['actual_mood'] == df['major_feeling']
        accuracy = df['is_correct'].mean()
    else:
        accuracy = None

    # Generate individual tables
    mood_tables = {}
    for mood in moods.keys():
        table = df[df['major_feeling'] == mood][['major_feeling', 'song', 'major_weight']]
        mood_tables[mood] = table.rename(columns={'major_weight': 'weight'}).sort_values(by='weight', ascending=False).reset_index(drop=True)

    return mood_tables, accuracy

# Streamlit UI
st.title("🎵 Mood-Based Song Tables")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    mood_tables, accuracy = generate_feeling_tables(uploaded_file)

    if accuracy is not None:
        st.success(f"🎯 Mood Prediction Accuracy: {accuracy:.2%}")
    else:
        st.warning("⚠️ No 'actual_mood' column found. Accuracy can't be calculated.")

    tabs = st.tabs(["😊 Happy", "😢 Sad", "💕 Romantic", "🧘 Relaxing", "⚡ Energetic"])
    mood_names = ["happy", "sad", "romantic", "relaxing", "energetic"]

    for i, mood in enumerate(mood_names):
        with tabs[i]:
            st.subheader(f"{mood.capitalize()} Songs")
            st.dataframe(mood_tables[mood])
else:
    st.info("Please upload a CSV file to see the mood-based song tables.")
