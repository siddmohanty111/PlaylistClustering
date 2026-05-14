import glob
import json
import os
import re
import unicodedata

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


DATA_DIR = "data"
OUTPUT_FILE = "no_norm_embeddings.pkl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dim
BATCH_SIZE = 1024
ENCODE_BATCH_SIZE = 512
MAX_JSON_FILES = None  
NORMALIZE_EMBEDDINGS = False  


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm > 0:
        return vec / norm
    return vec


def _normalize_title(title: str) -> str:
    text = unicodedata.normalize("NFKC", title)
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _process_batch(
    model: SentenceTransformer,
    titles: list[str],
    tracks_list: list[list[str]],
    track_cache: dict[str, np.ndarray],
):
    title_vecs = model.encode(
        titles,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
        show_progress_bar=False,
        batch_size=ENCODE_BATCH_SIZE,
    )

    unseen_tracks = list({t for tracks in tracks_list for t in tracks if t and t not in track_cache})
    if unseen_tracks:
        new_track_vecs = model.encode(
            unseen_tracks,
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
            show_progress_bar=False,
            batch_size=ENCODE_BATCH_SIZE,
        )
        for i, track in enumerate(unseen_tracks):
            track_cache[track] = new_track_vecs[i]

    dim = model.get_sentence_embedding_dimension()
    tracks_embeddings = []
    for tracks in tracks_list:
        if not tracks:
            mean_vec = np.zeros(dim, dtype=np.float32)
        else:
            vecs = [track_cache[t] for t in tracks if t in track_cache]
            if not vecs:
                mean_vec = np.zeros(dim, dtype=np.float32)
            else:
                mean_vec = np.mean(np.asarray(vecs, dtype=np.float32), axis=0)
                if NORMALIZE_EMBEDDINGS:
                    mean_vec = _l2_normalize(mean_vec)
        tracks_embeddings.append(mean_vec)

    return title_vecs, tracks_embeddings


def main() -> None:
    json_files = sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.json"), recursive=True))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found under `{DATA_DIR}`")
    if MAX_JSON_FILES is not None:
        json_files = json_files[:MAX_JSON_FILES]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    dim = model.get_sentence_embedding_dimension()
    print(f"Using device: {device}")
    print(f"Normalize embeddings: {NORMALIZE_EMBEDDINGS}")

    all_titles = []
    all_tracks = []
    all_title_embeddings = []
    all_tracks_embeddings = []
    track_cache: dict[str, np.ndarray] = {}

    batch_titles = []
    batch_tracks = []
    progress = tqdm(desc="Processing playlists", unit="playlist")

    for fp in tqdm(json_files, desc="Loading JSON files"):
        with open(fp, "r", encoding="utf-8") as f:
            chunk = json.load(f)

        for pl in chunk.get("playlists", []):
            title = _normalize_title(str(pl.get("name", "") or ""))
            tracks = []
            for t in pl.get("tracks", []):
                track_name = str(t.get("track_name", "") or "").strip()
                if track_name:
                    tracks.append(track_name)

            batch_titles.append(title)
            batch_tracks.append(tracks)

            if len(batch_titles) >= BATCH_SIZE:
                title_vecs, tracks_vecs = _process_batch(
                    model, batch_titles, batch_tracks, track_cache
                )
                all_titles.extend(batch_titles)
                all_tracks.extend(batch_tracks)
                all_title_embeddings.extend(title_vecs)
                all_tracks_embeddings.extend(tracks_vecs)
                progress.update(len(batch_titles))
                batch_titles, batch_tracks = [], []

    if batch_titles:
        title_vecs, tracks_vecs = _process_batch(
            model, batch_titles, batch_tracks, track_cache
        )
        all_titles.extend(batch_titles)
        all_tracks.extend(batch_tracks)
        all_title_embeddings.extend(title_vecs)
        all_tracks_embeddings.extend(tracks_vecs)
        progress.update(len(batch_titles))

    progress.close()

    df = pd.DataFrame(
        {
            "playlist_title": all_titles,
            "title_embedding": all_title_embeddings,
            "playlist_tracks": all_tracks,
            "tracks_embedding": all_tracks_embeddings,
        }
    )
    df.to_pickle(OUTPUT_FILE)

    print(f"Saved {OUTPUT_FILE}")
    print(f"Rows: {len(df)}")
    print(f"Embedding dim: {dim}")
    print(f"Unique tracks embedded: {len(track_cache)}")


if __name__ == "__main__":
    main()