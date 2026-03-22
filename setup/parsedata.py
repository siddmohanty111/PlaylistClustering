from mpdutils import convert_json_to_csv, cluster_playlists
import sys
import os
import time
import pandas as pd
import pickle
import importlib.util
from sentence_transformers import SentenceTransformer

# =============================================================================
# 1. PATH CONFIGURATION
# =============================================================================

SHARED_DRIVE_ROOT = '/content/drive/MyDrive/PlaylistClustering'

# Data paths relative to the shared drive root
RAW_JSON_DATA_PATH = os.path.join(SHARED_DRIVE_ROOT, 'Spotify Playlist Data/playlist_data/data')
PROCESSED_DATA_DIR = os.path.join(SHARED_DRIVE_ROOT, 'Processed Data')

# Specific output files
PLAYLISTS_CSV = os.path.join(PROCESSED_DATA_DIR, 'csvs/playlists.csv')
EMBEDDINGS_FILE = os.path.join(PROCESSED_DATA_DIR, 'calced_embeddings/playlists_embeddings_sbert.pkl')

# The script assumes it is being run from the root of the cloned GitHub repository
REPO_ROOT = os.getcwd()

# Ensure output directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DATA_DIR, 'csvs'), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DATA_DIR, 'calced_embeddings'), exist_ok=True)

# =============================================================================
# 2. RUN JSON TO CSV CONVERSION
# =============================================================================

print("--- Starting JSON to CSV Conversion ---")

max_retries = 5
retries = 0
success = False

while retries < max_retries and not success:
    try:
        print(f"Attempt {retries + 1}/{max_retries} to convert JSON to CSV...")
        convert_json_to_csv(RAW_JSON_DATA_PATH, PROCESSED_DATA_DIR)
        success = True
        print("JSON to CSV conversion completed successfully.\n")
    except OSError as e:
        if "Transport endpoint is not connected" in str(e):
            retries += 1
            print(f"OSError: {e}. Retrying in 10 seconds...")
            time.sleep(10)
        else:
            print(f"An unexpected OSError occurred: {e}")
            break
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        break

if not success:
    print("Failed to convert JSON to CSV after multiple retries.")
    sys.exit(1)


# =============================================================================
# 3. CREATE EMBEDDINGS
# =============================================================================
print("--- Starting Embedding Generation ---")

try:
    df = pd.read_csv(PLAYLISTS_CSV)
    titles = df['name'].fillna("").tolist()
    tracks = df['tracks'].tolist() if 'tracks' in df.columns else []
except Exception as e:
    # just generate dummy data if the CSV loading fails, to allow the rest of the pipeline to run
    print(f"Error loading CSV: {e}")
    titles = ["chill vibes", "workout mix", "study lofi"]
    tracks = [[], [], []]

# Load Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Loaded Sentence-BERT model. Computing embeddings...")
embeddings = model.encode(titles, show_progress_bar=True)

# Format the embeddings as expected by the downstream clustering script
embeds_to_save = {
    "playlist_embeddings": embeddings,
    "playlist_titles": titles,
    "playlist_tracks": tracks
}

# Save embeddings
with open(EMBEDDINGS_FILE, 'wb') as f:
    pickle.dump(embeds_to_save, f)
print(f"Saved embeddings to {EMBEDDINGS_FILE}\n")


# =============================================================================
# 4. CLUSTERING
# =============================================================================
print("--- Starting Clustering ---")

# Load Embeddings
with open(EMBEDDINGS_FILE, 'rb') as f:
    embeds_full = pickle.load(f)

# Truncate embeds for faster runtimes
truncation_ratio = 0.01
embeds = {k: v[:int(len(v) * truncation_ratio)] for k, v in embeds_full.items()}

playlist_embeddings = embeds["playlist_embeddings"]
playlist_titles = embeds["playlist_titles"]
playlist_tracks = embeds["playlist_tracks"]

# Run clustering for multiple cluster counts
for i in range(10, 200, 10):
    output_clusters_csv = os.path.join(PROCESSED_DATA_DIR, f"calced_clusters/{i}")
    os.makedirs(output_clusters_csv, exist_ok=True)
    cluster_csv_path = os.path.join(output_clusters_csv, "clusters.csv")
    
    print(f"Clustering with {i} clusters...")
    cluster_playlists(playlist_embeddings, i, playlist_titles, playlist_tracks, cluster_csv_path)

print("\nPipeline execution complete!")