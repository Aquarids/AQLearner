from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

import os
from tqdm import tqdm

from .base_rag import BaseRAG

class ContrieverRAG(BaseRAG):
    def __init__(self, dir: str, model_name="facebook/contriever"):
        super().__init__(dir)
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vector_store = None
        self.index_path = os.path.join(dir, "contriever_faiss")
        

    def retrieve_fact(self, query: str, max_len=4096, k=10):
        if not self.vector_store:
            raise ValueError("Index not loaded")
        
        docs = self._retrieve_documents(query, k=k)
        print(docs)
        docs = self._compress_documents(docs, max_len)
        return docs

    def _retrieve_documents(self, query: str, k: int):
        return self.vector_store.similarity_search(query, k=k)

    def build_index(self, documents):
        split_docs = self._split_documents(documents)
        self.vector_store = FAISS.from_documents(
            split_docs,
            self.embeddings
        )
        self.vector_store.save_local(self.index_path)
        return len(split_docs)

    def load_index(self):
        self.vector_store = FAISS.load_local(
            self.index_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    def validate_dataset_document(doc: Document):
        required_metadata = ["sample_id", "source"]
        return all(key in doc.metadata for key in required_metadata) and len(doc.page_content) > 50