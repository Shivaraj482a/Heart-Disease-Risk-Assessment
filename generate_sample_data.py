# Copy and run this complete code in a new Colab cell
from google.colab import drive
import pandas as pd
import numpy as np
import os

# Force remount the drive
drive.mount('/content/drive', force_remount=True)

# Create sample data
print("Creating sample data...")
data = {
    'age': [63, 67, 45, 52, 58],
    'sex': [1, 0, 1, 1, 0],
    'cp': [3, 2, 1, 2, 3],
    'trestbps': [145, 160, 130, 142, 132],
    'chol': [233, 286, 219, 225, 197],
    'fbs': [1, 0, 0, 0, 1],
    'restecg': [0, 0, 1, 1, 1],
    'thalach': [150, 108, 168, 155, 173],
    'exang': [0, 1, 0, 0, 0],
    'oldpeak': [2.3, 1.5, 1.8, 2.0, 1.6],
    'slope': [0, 1, 2, 1, 0],
    'ca': [0, 3, 0, 2, 1],
    'thal': [1, 2, 3, 2, 2]
}

# Create DataFrame
df = pd.DataFrame(data)

# Define the save path
save_path = '/content/drive/MyDrive/heart_test_data.csv'

# Save the file
try:
    df.to_csv(save_path, index=False)
    print(f"\nAttempting to save file to: {save_path}")
    
    # Verify file exists
    if os.path.exists(save_path):
        print("✓ File successfully saved and verified!")
        print("\nFile contents:")
        print(pd.read_csv(save_path).head())
    else:
        print("✗ File was not saved successfully.")
        
    # List all files in the directory
    print("\nFiles in MyDrive directory:")
    files = os.listdir('/content/drive/MyDrive')
    for file in files:
        if file.endswith('.csv'):
            print(f"- {file}")
            
except Exception as e:
    print(f"Error: {str(e)}")

# Double check file existence
print("\nFinal verification:")
print(f"File exists: {os.path.exists(save_path)}") 