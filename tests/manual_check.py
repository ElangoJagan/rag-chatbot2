from src.ingestion.document import Document, make_doc_id
d1 = Document(doc_id=make_doc_id("hello"), source="test", text="hello")
d2 = Document(doc_id=make_doc_id("world"), source="test", text="world")
d1.metadata["dept"] = "HR"
print("d2 metadata (should be empty):", d2.metadata)
print(d1)