import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Load dataset
df = pd.read_csv('dataset/dataset.csv')

# Basic statistics
print(f"Total samples: {len(df)}")
print(f"Class distribution:\n{df['label'].value_counts()}")

# Text length analysis
df['text_length'] = df['text'].str.len()
print(f"\nText length statistics:")
print(f"Min: {df['text_length'].min()}")
print(f"Max: {df['text_length'].max()}")
print(f"Mean: {df['text_length'].mean():.1f}")
print(f"Median: {df['text_length'].median():.1f}")

# Word count analysis
df['word_count'] = df['text'].str.split().str.len()
print(f"\nWord count statistics:")
print(f"Min: {df['word_count'].min()}")
print(f"Max: {df['word_count'].max()}")
print(f"Mean: {df['word_count'].mean():.1f}")
print(f"Median: {df['word_count'].median():.1f}")

# Most common words by class
def get_top_words(texts, n=10):
    words = ' '.join(texts).lower().split()
    word_counts = Counter(words)
    return word_counts.most_common(n)

print("\nTop words in Left-leaning texts:")
print(get_top_words(df[df['label'] == 'Left']['text']))

print("\nTop words in Center-leaning texts:")
print(get_top_words(df[df['label'] == 'Center']['text']))

print("\nTop words in Right-leaning texts:")
print(get_top_words(df[df['label'] == 'Right']['text']))

# Visualize class distribution
df['label'].value_counts().plot(kind='bar', color=['blue', 'green', 'red'])
plt.title('Class Distribution')
plt.xlabel('Political Leaning')
plt.ylabel('Count')
plt.savefig('dataset/class_distribution.png')
plt.close()

# Visualize text length distribution by class
df.boxplot(column='text_length', by='label', grid=False)
plt.title('Text Length Distribution by Political Leaning')
plt.suptitle('')
plt.xlabel('Political Leaning')
plt.ylabel('Text Length')
plt.savefig('dataset/text_length_distribution.png')
plt.close()

print("\nValidation complete. Visualizations saved to dataset directory.")