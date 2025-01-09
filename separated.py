import os
import shutil

# Define the path to the Extracted_Frames folder and the new folder to store separated classes
extracted_frames_directory = 'Extracted_Frames'
output_directory = 'Separated_Frames'

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Define the classes and initialize dictionaries to store frame filenames for each class
classes = ['Boredom', 'Confusion', 'Engagement', 'Frustrated', 'Sleepy', 'Yawning']
frames_per_class = 6501

# Dictionary to store frames for each class
class_frames = {cls: [] for cls in classes}

# Step 1: Organize frames by class
for file_name in os.listdir(extracted_frames_directory):
    # Assuming the filenames include the class label as a prefix (e.g., Sleepy_Sleepy38_frame114.jpg)
    for cls in classes:
        if file_name.startswith(cls):
            class_frames[cls].append(file_name)
            break

# Step 2: Create subfolders for each class and move the files
for cls in classes:
    class_folder = os.path.join(output_directory, cls)
    
    # Create the subfolder for the class if it doesn't exist
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)
    
    # Get the first 6501 frames for the class
    selected_frames = class_frames[cls][:frames_per_class]
    
    # Step 3: Rename and move the frames into the corresponding class folder
    for index, file_name in enumerate(selected_frames):
        # Extract the frame number (this assumes the filename has a pattern like ClassName_FrameNumber.jpg)
        new_filename = f"{cls}_{cls}{index+1}_frame{index+1}.jpg"
        
        # Define the source and destination paths
        src_path = os.path.join(extracted_frames_directory, file_name)
        dest_path = os.path.join(class_folder, new_filename)
        
        # Move and rename the file
        shutil.copy(src_path, dest_path)
    
    print(f"{len(selected_frames)} frames for class {cls} have been moved and renamed.")

print("\nProcess completed. All classes have been separated and stored in the 'Separated_Frames' directory.")
