import numpy as np
from sentence_transformers import SentenceTransformer

# Load cleaned data
def load_cleaned_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

# Generate embeddings for the cleaned data
def generate_embeddings(data):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(data, convert_to_tensor=True)
    return embeddings

# Save embeddings to a file
def save_embeddings(embeddings, output_file):
    np.save(output_file, embeddings)

# Main function
if __name__ == "__main__":
    cleaned_file_path = 'cleaned_data.txt'
    output_file_path = 'embeddings.npy'
    
    data = load_cleaned_data(cleaned_file_path)
    embeddings = generate_embeddings(data)
    save_embeddings(embeddings, output_file_path)
    
    print(f"Embeddings saved to {output_file_path}")
