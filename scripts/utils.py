import os

def save_processed_data(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data.to_csv(path, index=False)
    print(f"Data saved to {path}")
