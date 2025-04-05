import os
import pandas as pd
import matplotlib.pyplot as plt

# Define function to count words in a file
def count_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        return len(text.split())

# Base directory containing your transcripts folders
base_dir = "/Users/robbie/Downloads/transcripts"

# List to store word count data for each file
data = []

# Loop through each folder ending with "_transcripts"
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    if os.path.isdir(folder_path) and folder_name.endswith("_transcripts"):
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                try:
                    word_count = count_words(file_path)
                    # Extract ticker from folder name (e.g., AAPL_transcripts → AAPL)
                    ticker = folder_name.replace("_transcripts", "")
                    data.append({"ticker": ticker, "filename": filename, "word_count": word_count})
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Calculate summary statistics using pandas
summary_df = pd.DataFrame({
    "Average": [df["word_count"].mean()],
    "Minimum": [df["word_count"].min()],
    "Maximum": [df["word_count"].max()],
    "Standard Deviation": [df["word_count"].std()],
    "Variance": [df["word_count"].var()],
    "File Count": [len(df)],
    "Files > 12,000z": [df[df["word_count"] > 12000].shape[0]],
})

# Display the summary DataFrame
print("Transcript Word Count Summary:")
print(summary_df)

# Create a histogram of word counts using bins of 500-word width
max_word_count = df["word_count"].max()
bins = range(0, (max_word_count // 500 + 2) * 500, 500)

plt.figure(figsize=(10, 6))
plt.hist(df["word_count"], bins=bins, edgecolor='black')
plt.title("Distribution of Word Counts in Transcripts")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.xticks(bins, rotation=45)
plt.grid(True)
plt.show()
