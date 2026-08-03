from dataclasses import dataclass, field
import hashlib

@dataclass
class Document:
    doc_id:str
    source:str
    text:str 
    metadata:dict = field(default_factory = dict)

def make_doc_id(text:str):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]