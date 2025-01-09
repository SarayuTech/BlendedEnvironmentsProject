import os
import pandas as pd

# Load the balanced dataset
balanced_data = pd.read_csv('balanced_frames.csv')

# Extract the filenames and labels from the balanced dataset
balanced_filenames = set(balanced_data['feature'].apply(lambda x: os.path.basename(x)))
classification_counts = balanced_data['label'].value_counts()

# Print the classification counts from the balanced dataset
print("Classification counts from 'balanced_frames.csv':")
print(classification_counts)

# Define the directory containing the original frames
frames_directory = 'Extracted_Frames'

# Count the number of matched frames for each classification in the directory
matched_classifications = {label: 0 for label in balanced_data['label'].unique()}

for file_name in os.listdir(frames_directory):
    
    if file_name in balanced_filenames:
        # Find the corresponding classification
        label = balanced_data[balanced_data['feature'].str.contains(file_name)]['label'].values[0]
        matched_classifications[label] += 1

# Print the matched classification counts
print("\nCount of matched frames from 'Extracted_Frames':")
for label, count in matched_classifications.items():
    print(f"{label}: {count}")
