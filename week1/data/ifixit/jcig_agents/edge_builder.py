from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class EdgeBuilderAgent:
    @staticmethod
    def run(concepts):
        embs = [np.mean([concepts['embeddings'][i] for i in inds], axis=0) for inds in concepts['groups']]
        sim = cosine_similarity(embs)
        edges = []
        for i, row in enumerate(sim):
            sorted_sim = sorted([(j, score) for j, score in enumerate(row) if j != i], key=lambda x: -x[1])
            if sorted_sim:
                j, score = sorted_sim[0]
                edges.append((i, j, round(score, 2)))
        return edges