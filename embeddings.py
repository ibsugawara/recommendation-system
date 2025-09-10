from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

animes = pd.read_csv("artifacts/anime_list.csv")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(animes['tags'].tolist(), convert_to_tensor=False)

np.save("artifacts/anime_embeddings.npy", embeddings)