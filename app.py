import pandas as pd

df = pd.read_csv('test data.csv')

# Clean up column names
df.columns = df.columns.str.strip().str.lower()

# Check what's there
print("Available columns:", df.columns.tolist())

# Your emotion columns
emotion_columns = ['angry', 'happy', 'calm', 'sad', 'fear']

# Make sure all required columns exist
missing = [col for col in emotion_columns if col not in df.columns]
if missing:
    print(f"Missing columns: {missing}")
else:
    df['major_feeling'] = df[emotion_columns].idxmax(axis=1)
