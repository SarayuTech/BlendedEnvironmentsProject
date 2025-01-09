import pandas as pd
from sklearn.utils import resample

# Load the dataset
file_path = 'frames.csv'
data = pd.read_csv(file_path)

# Determine the minimum number of samples among all classes
min_samples = data['label'].value_counts().min()

# Downsample each class to have the same number of samples
balanced_data = pd.concat([
    resample(data[data['label'] == label], n_samples=min_samples, random_state=42)
    for label in data['label'].unique()
])

# Shuffle the balanced dataset
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the balanced dataset to a new CSV file
balanced_data.to_csv('balanced_frames.csv', index=False)

print("Balanced dataset created and saved as 'balanced_frames.csv'.")
