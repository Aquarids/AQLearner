from langchain_community.document_loaders import WikipediaLoader
import os

from .base_rag import BaseRAG

class WikiRAG(BaseRAG):
    def __init__(self, dir: str):
        super().__init__(dir)
        self.wiki_cache_path = os.path.join(dir, "wiki_cache.db")

    def retrieve_fact(self, query: str, lang: str = "zh", max_len: int = 4096, top_k: int = 3):
        docs = self._retrieve_documents(query, lang=lang, top_k=top_k)
        split_docs = self._split_documents(docs)
        return self._compress_documents(split_docs, max_len)

    def _retrieve_documents(self, query: str, lang: str, top_k: int):
        loader = WikipediaLoader(
            query=query,
            lang=lang,
            load_max_docs=top_k,
            doc_content_chars_max=4000
        )
        return loader.load()
