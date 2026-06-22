from sentence_transformers import SentenceTransformer
import numpy as np, os
from lib.search_utils import MOVIE_EMBEDDINGS_PATH, load_movies

class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}
    
    def generate_embedding(self, text) -> str:
        if text == None:
            raise ValueError
        embedding = self.model.encode([text])

        return embedding[0]
    
    def build_embeddings(self, documents) -> list[float]:
        self.documents = documents
        embedding_list = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            embedding_list.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(embedding_list,show_progress_bar=True)
        np.save(MOVIE_EMBEDDINGS_PATH,self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        if os.path.exists(MOVIE_EMBEDDINGS_PATH):
          self.embeddings = np.load(MOVIE_EMBEDDINGS_PATH)
          if len(self.embeddings) == len(documents):
              return self.embeddings
        return self.build_embeddings(documents)

            

def verify_model():
    search_instance = SemanticSearch()
    print(f"Model loaded: {search_instance.model}")
    print(f"Max sequence length: {search_instance.model.max_seq_length}")

def embed_text(text) -> str:
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    search_instance = SemanticSearch()
    movies = load_movies()
    search_instance.load_or_create_embeddings(movies)
    print(f"Number of docs:   {len(search_instance.document_map)}")
    print(f"Embeddings shape: {search_instance.embeddings.shape[0]} vectors in {search_instance.embeddings.shape[1]} dimensions")   
    

    
