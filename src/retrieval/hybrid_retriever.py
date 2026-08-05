import numpy as np 
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.processing.chunking import Chunk
from src.embeddings.embedder import Embedder, FAISSVectorStore

class HybridRetriever:
    def __init__(self, vector_score, embedder, chunks, vector_weight=0.6, bm25_weight=0.4, reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.vector_score = vector_score
        self.embedder = embedder
        self.chunks = chunks
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        
        tokenized_corpus = [c.text.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.reranker = CrossEncoder(reranker_model)
        
    
    def retrieve(self, query, top_k= 5, candidate_pool =15):
        
        #1.Vector store
        query_emb = self.embedder.embed([query])[0]
        vector_results = self.vector_score.search(query_emb, top_k = candidate_pool)
        vector_stores = {chunk.chunk_id: score for chunk, score in vector_results}
        
        #2. bm25 Keyword search
        bm25_scores_all = self.bm25.get_scores(query.lower().split())
        top_bm25_idx = np.argsort(bm25_scores_all)[::-1][:candidate_pool]
        bm25_scores = {self.chunks[i].chunk_id:bm25_scores_all[i] for i in top_bm25_idx}
        
        
        #3.fuse both 
        all_ids = set(vector_stores) | set(bm25_scores)
        fused = self._fuse(all_ids, vector_stores, bm25_scores)
        
        
        candidate_ids = [cid for cid , _ in sorted(fused.items(), key=lambda x: -x[1])[:candidate_pool]]
        chunk_by_id = {c.chunk_id: c for  c in  self.chunks}
        candidates = [chunk_by_id[cid] for cid in candidate_ids]
        
        #4. re-rank the fused  candidates  with a cross-encoder
        pairs = [(query, c.text) for c in candidates]
        rerank_scores =self.reranker.predict(pairs)
        
        ranked = sorted(zip(candidates, rerank_scores ), key = lambda x: -x[1])[:top_k]
        return [(chunk, float(score)) for chunk, score in ranked]
    
    def _fuse(self, all_ids, vec_scores, bm25_scores):
        def normalize(d):
            if not d:
                return {}
            vals = list(d.values())
            lo, hi = min(vals), max(vals)
            if hi == lo:
                return {k: 1.0 for k in d}
            return {k: (v-lo)/(hi - lo) for k, v in d.items()}
        vec_n,bm25_n = normalize(vec_scores), normalize(bm25_scores)
        return{
            cid:self.vector_weight*vec_n.get(cid, 0)+self.bm25_weight * bm25_n.get(cid,0)
            for cid in all_ids
        }
        