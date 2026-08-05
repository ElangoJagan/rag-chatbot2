from src.ingestion.file_loader import FileLoader
from src.processing.chunking import SemanticChunker
from src.embeddings.embedder import Embedder, FAISSVectorStore
from src.retrieval.hybrid_retriever import HybridRetriever


loader = FileLoader('data/raw')
chunker   = SemanticChunker(max_chunk_size = 500, overlap = 50)


all_chunks= []
for doc in loader.load():
    
    chunks = chunker.chunk_document(doc)
    all_chunks.extend(chunks)
    print(doc.source, "->", len(chunks), "chunks")

print(all_chunks)

embedder = Embedder()
texts = [c.text for c in all_chunks]
embeddings = embedder.embed(texts)
print("Embeddings shape:", embeddings.shape)

store = FAISSVectorStore(dim = 384)
store.add(all_chunks, embeddings)
    
query = 'How many vacation  days do employee get ?'
query_emb_1= embedder.embed([query])
print(f'query emb {query_emb_1}')
query_emb= embedder.embed([query])[0]
results = store.search(query_emb, top_k =3)



retriever = HybridRetriever(store, embedder, all_chunks)
results = retriever.retrieve("How many vacation days do employees get?", top_k=3)

for chunk , score in results:
    print(f"\nScore: {score:.3f}")
    print(f'text : {chunk.text[:150]}')

results = retriever.retrieve("What is the capital of France?", top_k=3)
for chunk, score in results:
    print(f"Score: {score:.3f}")