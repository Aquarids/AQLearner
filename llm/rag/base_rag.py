from abc import ABC, abstractmethod
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

class BaseRAG(ABC):
    def __init__(self, dir: str, chunk_size=512, chunk_overlap=64):
        self.cache_dir = dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n===", "\n##", "\n\n", "\n", "。"]
        )
        self._init_cache()

    def _init_cache(self):
        cache_path = os.path.join(self.cache_dir, "rag_cache.db")
        if not os.path.exists(cache_path):
            os.makedirs(self.cache_dir, exist_ok=True)

        set_llm_cache(SQLiteCache(database_path=cache_path))

    def retrieve_fact(self, query: str, **kwargs):
        raise NotImplementedError("Not implemented")

    def _retrieve_documents(self, query: str, **kwargs):
        raise NotImplementedError("Not implemented")

    def _split_documents(self, docs):
        return self.text_splitter.split_documents(
            [doc for doc in docs if not doc.page_content.startswith("Redirect")]
        )

    def _compress_documents(self, docs, max_len=4096):
        compressed = []
        total_len = 0
        for doc in docs:
            content = doc.page_content
            if total_len + len(content) > max_len:
                remaining = max_len - total_len
                compressed.append(content[:remaining])
                break
            compressed.append(content)
            total_len += len(content)
        return "\n\n".join(compressed)
