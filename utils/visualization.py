import matplotlib.pyplot as plt
import seaborn as sns

def plot_distribution(df, title="Emotion Distribution"):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='emotion', hue='source', palette='Set2')
    plt.title(title)
    plt.xlabel('Emotions')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
