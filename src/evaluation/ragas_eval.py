from sklearn.metrics.pairwise import cosine_similarity

def semantic_score(embedder, pred, truth):
    try:
        emb = embedder.encode([pred, truth])
        return cosine_similarity([emb[0]], [emb[1]])[0][0]
    except:
        return 0.0