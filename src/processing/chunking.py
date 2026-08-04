from dataclasses import dataclass, field
import re

from src.ingestion.document import Document

@dataclass
class Chunk:
    chunk_id:str
    doc_id:str
    text:str
    chunk_index:int
    source:str
    metadata:dict = field(default_factory =dict)

class SemanticChunker:
    def __init__(self, max_chunk_size = 500, overlap = 50, min_chunk_size = 100):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        
    def _split_paragraphs(self,text):
        paras = re.split(r'\n\s*\n',text)
        return [p.strip() for p in paras if p.strip()]
    
    def _pack_paragraphs(self, paragraphs):
        chunks = []
        current =''
        for para in paragraphs:
            if len(current)+len(para)+1 <= self.max_chunk_size:
                current = f'{current}\n{para}'if current else para
            else:
                if current:
                    chunks.append(current)
                overlap_text = current[-self.overlap:] if current else ""
                current =f'{overlap_text}\n{para}'.strip() if overlap_text else para
        if current:
            chunks.append(current)
        return chunks
    
    def _merge_small_chunks(self,raw_chunks):
        if not raw_chunks:
            return raw_chunks
        merged = []
        for text in raw_chunks:
            if merged and len(text.strip())<self.max_chunk_size:
                merged[-1] = f'{merged[-1]}\n{text}'.strip()
            else:
                merged.append(text)
        if len(merged)>1 and len(merged[0].strip())<self.min_chunk_size:
            merged[1] = f'{merged[0]}\n{merged[1]}'.strip()
            merged = merged[1:]
        return merged
    
    def chunk_document(self,doc):
        paragraphs = self._split_paragraphs(doc.text)
        raw_chunks = self._pack_paragraphs(paragraphs)
        raw_chunks = self._merge_small_chunks(raw_chunks)
        
        chunks = []
        for i , text in enumerate(raw_chunks):
            if not text.strip():
                continue
            chunks.append(Chunk(
                chunk_id = f'{doc.doc_id}_{i}',
                doc_id = doc.doc_id,
                text = text.strip(),
                chunk_index = i,
                source = doc.source,
                metadata = doc.metadata
            ))
        return chunks 
    
        