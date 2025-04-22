import shutil
from pathlib import Path

# Source directory (Google Drive)
drive_path = Path("/content/drive/MyDrive")

# Destination directory (repository)
repo_path = Path("models")

# Create models directory if it doesn't exist
repo_path.mkdir(exist_ok=True)

# List of all model files
model_files = [
    "grid_search_best_model.joblib",
    "decision_tree_3_gini.joblib",
    "decision_tree_4_gini.joblib",
    "decision_tree_5_entropy.joblib",
    "decision_tree_7_gini.joblib",
    "decision_tree_None_entropy.joblib",
    "random_forest_100_5.joblib",
    "random_forest_150_6.joblib",
    "random_forest_200_7.joblib",
    "random_forest_200_None.joblib",
    "random_forest_300_10.joblib",
    "gradient_boosting_100_0.1.joblib",
    "gradient_boosting_150_0.1.joblib",
    "gradient_boosting_200_0.05.joblib",
    "gradient_boosting_250_0.05.joblib",
    "gradient_boosting_300_0.01.joblib",
    "adaboost_50_1.0.joblib",
    "adaboost_100_0.5.joblib",
    "adaboost_150_0.8.joblib",
    "adaboost_200_0.1.joblib",
    "adaboost_300_0.05.joblib"
]

# Copy each model file
for model_file in model_files:
    source = drive_path / model_file
    destination = repo_path / model_file
    if source.exists():
        print(f"Copying {model_file}...")
        shutil.copy2(source, destination)
        print(f"Successfully copied {model_file}")
    else:
        print(f"Warning: {model_file} not found in drive")

print("\nAll models have been copied to the repository's models directory.") 