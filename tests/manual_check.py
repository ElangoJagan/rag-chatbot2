from src.ingestion.file_loader import FileLoader
from src.processing.chunking import SemanticChunker
from src.embeddings.embedder import Embedder, FAISSVectorStore


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

for chunk , score in results:
    print(f"\nScore: {score:.3f}")
    print(f'text : {chunk.text[:150]}')
