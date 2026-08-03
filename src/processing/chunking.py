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
    
    def _merge_small_chunks(self,raw_chunks)