import numpy as np
from sentence_transformers import SentenceTransformer

# Function to load cleaned data
def load_cleaned_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

# Function to load embeddings from file
def load_embeddings(file_path):
    return np.load(file_path)

# Test loading of files
if __name__ == '__main__':
    try:
        # Load precomputed data and embeddings
        data = load_cleaned_data('cleaned_data.txt')
        embeddings = load_embeddings('embeddings.npy')
        
        # Verify loaded data
        print(f"Loaded cleaned data with {len(data)} lines.")
        print(f"Loaded embeddings with shape {embeddings.shape}.")
        
        # Optionally, you can print a sample to verify contents
        print("Sample data:", data[:5])
        print("Sample embedding:", embeddings[0])
        
    except Exception as e:
        print(f"An error occurred: {e}")
