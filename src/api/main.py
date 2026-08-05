from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq

from src.ingestion.file_loader import FileLoader
from src.processing.chunking import SemanticChunker
from src.retrieval.hybrid_retriever import HybridRetriever
from src.embeddings.embedder import Embedder, FAISSVectorStore


app = FastAPI(title = 'HR RAG ChatBot')

loader = FileLoader('data/raw')
chunker = SemanticChunker(max_chunk_size = 500, overlap = 50)

all_chunks = []
for doc in loader.load():
    all_chunks.extend(chunker.chunk_document(doc))

embedder = Embedder()
embeddings = embedder.embed([c.text for c in all_chunks])

store = FAISSVectorStore(dim = 384)
store.add(all_chunks, embeddings)

retriever = HybridRetriever(store, embedder, all_chunks)
llm_client = Groq()

class QueryRequest(BaseModel):
    question:str
    
@app.get('/health')
def health():
    return {"status": "ok", "indexed_chunks": len(all_chunks)}


CONFIDENCE_THRESHOLD = 0.0

@app.post('/query')
def query(req:QueryRequest):
    results = retriever.retrieve(req.question, top_k =5)
    
    top_score = results [0][1] if results else -999
    
    top_score = results[0][1] if results else -999
    print(f"DEBUG top_score: {top_score}")
    
    if top_score < CONFIDENCE_THRESHOLD:
        return {
            'answer': "I don't have enough relevant information in the HR handbook to answer that question confidently.",
            'sources':[],
            'confidence' : top_score
        }
    
    context = '\n---\n'.join(chunk.text for chunk, score in results)
    sources = list({chunk.source for chunk,score in results})
    
    response = llm_client.chat.completions.create(
        model= "llama-3.3-70b-versatile",
        max_tokens = 500,
        messages = [{
            'role':'user',
            'content':f"Context:\n {context}\n\nQuestion: {req.question}\n\nAnswer using only the context above."
            
        }]
    )
    
    return{
        "answer": response.choices[0].message.content,
        "sources": sources
    }