import os, numpy as np

class LiteEmbedder:
    """Local dense embedder: char n-gram TF-IDF -> LSA (SVD). No downloads."""
    def __init__(self, dim=128):
        self.dim = dim
    def fit(self, texts):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        self.tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), sublinear_tf=True, lowercase=True, max_features=15000, dtype=np.float32)
        X = self.tf.fit_transform(texts)
        k = max(2, min(self.dim, X.shape[0] - 1, X.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=k, n_iter=3, random_state=0).fit(X)
        return self
    def encode(self, texts):
        V = self.svd.transform(self.tf.transform(texts))
        n = np.linalg.norm(V, axis=1, keepdims=True); n[n == 0] = 1
        return V / n

class STEmbedder:
    def __init__(self, name="BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(name)
    def fit(self, texts): return self
    def encode(self, texts):
        return np.asarray(self.m.encode(texts, normalize_embeddings=True, show_progress_bar=False))

def get_embedder():
    if os.getenv("EMBED_BACKEND", "lite") == "st":
        try: return STEmbedder()
        except Exception as e: print("ST embedder unavailable, falling back:", e)
    return LiteEmbedder()
