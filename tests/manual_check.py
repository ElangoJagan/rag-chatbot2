from src.ingestion.file_loader import FileLoader
from src.processing.chunking import SemanticChunker

loader = FileLoader('data/raw')
chunker   = SemanticChunker(max_chunk_size = 500, overlap = 50)


all_chunks= []
for doc in loader.load():
    
    chunks = chunker.chunk_document(doc)
    all_chunks.extend(chunks)
    print(doc.source, "->", len(chunks), "chunks")

print("Total chunks:", len(all_chunks))
print("Sample chunk text:", all_chunks[0].text[:200])
    
