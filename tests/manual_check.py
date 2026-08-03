from src.ingestion.file_loader import FileLoader

loader = FileLoader('data/raw')
for doc in loader.load():
    print(doc.source, "-", len(doc.text), "characters")
