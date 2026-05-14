import re
import pickle
import unicodedata
import numpy as np
from sentence_transformers import SentenceTransformer
import random
import matplotlib.pyplot as plt

# remove white space and lowercase
def normalize_phrase(phrase: str):
    text = unicodedata.normalize("NFKC", phrase)
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text

# return stats about similar embeddings
def embedding_similarity(pkl_file: str, phrase: str, embedding_type: str = "tracks", topk_playlists: int = 5, topk_songs: int = 10):

    phrase = normalize_phrase(phrase)
    query_embedding = embed_phrase(phrase)

    data = None
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)

    embedding = None
    if embedding_type not in ["tracks", "titles"]:
        raise ValueError("No")
    if embedding_type == "tracks":
        embeddings = data["tracks_embedding"]
    if embedding_type == "titles":
        embeddings = data["title_embedding"]
    
    # cosine similarity ordering
    scores = {}
    for pid, embedding in embeddings.items():
        embedding = np.asarray(embedding, dtype=np.float32)
        scores[pid] = float(np.dot(query_embedding, embedding))

    # choose top k similar embeddings
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk_playlists]
    sims = [val[1] for val in top]
    playlist_indices = [val[0] for val in top]

    # get tracks from similar embeddings
    tracks = []
    titles = []
    for playlist in top: 
        tracks.extend(data.iloc[playlist[0]]["playlist_tracks"])
        titles.append(data.iloc[playlist[0]]["playlist_title"])

    # shuffle tracks 
    random.shuffle(tracks)
    selected_tracks = tracks[:topk_songs]

    return {
        "queried_title": phrase,
        "recommended_songs": selected_tracks,
        "embedding_type": embedding_type,
        "similar_playlists": titles,
        "similaries": sims,
        "similar_playlist_indices": playlist_indices,
        "topk_playlists": topk_playlists,
        "topk_songs": topk_songs,
    }

# lil wrapper to get embedding
def embed_phrase(phrase: str):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding = model.encode(phrase, normalize_embeddings=True)
    return embedding

# plot results
def plot_results(results: dict):
    import matplotlib.pyplot as plt

    labels = results.get("similar_playlists", [])
    scores = results.get("similaries", [])
    idxs = results.get("similar_playlist_indices", list(range(len(labels))))

    if not labels or not scores or len(labels) != len(scores):
        raise ValueError("Expected matching `similar_playlists` and `similaries`.")

    # Make x labels unique
    xlabels = [f"{i}: {t}" for i, t in zip(idxs, labels)]

    plt.figure(figsize=(14, 7))
    bars = plt.bar(range(len(scores)), scores)
    plt.xticks(range(len(scores)), xlabels, rotation=0, ha="center", fontsize=9)
    plt.ylabel("Cosine Similarity")
    plt.xlabel(f"Playlist (index: {results["embedding_type"]})")
    plt.title(f"Top Similar Playlists for: {results.get('queried_title', '')}")

    for b, s in zip(bars, scores):
        plt.text(b.get_x() + b.get_width()/2, b.get_height(), f"{s:.3f}",
                 ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.show()
    


if __name__ == "__main__":
    res = embedding_similarity("embeddings.pkl", "sad rock", "titles", 5, 10)
    print(res)
    plot_results(res)