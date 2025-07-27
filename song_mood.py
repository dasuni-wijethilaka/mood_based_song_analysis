import pandas as pd
import re

# ---------- Keyword matcher ----------
def contains_keywords(text, keywords):
    return bool(re.search(rf'\b({keywords})\b', str(text), re.IGNORECASE))

# ---------- Main mood analyzer ----------
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
        for mood in mood_scores:
            if mood == 'sad':
                weighted_scores[mood] = (
                    int(energy < 0.35) * 0.3 +
                    int(contains_keywords(lyrics, "sorry|cry|lonely|sad|goodbye|fall|apart|miss|nothing|lasts|feel so")) * 0.4 +
                    int(tempo < 85) * 0.2 +
                    int(key in [2, 3] and energy < 0.4) * 0.1
                )
            elif mood == 'happy':
                weighted_scores[mood] = (
                    int(energy > 0.75) * 0.3 +
                    int(contains_keywords(lyrics, "party|smile|sunshine|fun|make me smile|all night")) * 0.4 +
                    int(tempo > 120) * 0.2 +
                    int(key in [4, 7, 9]) * 0.1
                )
            elif mood == 'energetic':
                weighted_scores[mood] = (
                    int(energy > 0.8) * 0.35 +
                    int(contains_keywords(lyrics, "alive|fire|wild|rush|inside|shake")) * 0.35 +
                    int(tempo > 125) * 0.25 +
                    int(key in [4]) * 0.05
                )
            elif mood == 'romantic':
                weighted_scores[mood] = (
                    int(0.45 <= energy <= 0.65) * 0.2 +
                    int(contains_keywords(lyrics, "love|heart|kiss|baby|darling|hold|you|mine|beats")) * 0.5 +
                    int(80 <= tempo <= 100) * 0.2 +
                    int(key in [0, 2, 9]) * 0.1
                )
            elif mood == 'relaxing':
                weighted_scores[mood] = (
                    int(energy < 0.4) * 0.3 +
                    int(contains_keywords(lyrics, "sky|fall|good|calm|peace|slow|goodbye") and not contains_keywords(lyrics, "cry|sad|sorry")) * 0.3 +
                    int(70 <= tempo <= 90) * 0.3 +
                    int(key in [2, 3, 7]) * 0.1
                )

        sorted_moods = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
        major = sorted_moods[0][0]
        second = sorted_moods[1][0]

        results.append({
            'song': row['song'],
            'singer': row.get('singer', ''),
            'tempo': tempo,
            'energy': energy,
            **{f'{m}_weight': round(w, 3) for m, w in weighted_scores.items()},
            'major_feeling': major,
            'second_major_feeling': second
        })

    return pd.DataFrame(results)

# ---------- Run script ----------
if __name__ == "__main__":
    input_path = "preprocessed_test_data.csv"      # 👈 replace with your actual CSV file path
    output_path = "predicted_moods.csv"

    try:
        df = pd.read_csv(input_path)
        result_df = analyze_moods(df)
        result_df.to_csv(output_path, index=False)
        print(f"✅ Prediction completed! Results saved to: {output_path}")
    except Exception as e:
        print(f"❌ Error: {e}")
