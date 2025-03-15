from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.cache import SQLiteCache
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.globals import set_llm_cache

import os

class RAG:

    def __init__(self, dir, max_context=4096):
        self.max_context = max_context

        if not os.path.exists(dir):
            os.makedirs(dir)
        wiki_cache_path = os.path.join(dir, "wiki_cache.db")

        set_llm_cache(SQLiteCache(
            database_path=wiki_cache_path
        ))
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=64,
            separators=["\n===", "\n##", "\n\n", "\n", "。"]
        )

    def retrieve_fact(self, query: str, lang: str = "zh"):
        docs = self._wiki_retriever(query, lang)
        docs = self._split(docs)
        return self._compress_docs(docs)  
        
    def _wiki_retriever(self, query: str, lang: str = "zh", top_k: int = 3):
        loader = WikipediaLoader(
            query=query,
            lang=lang,
            load_max_docs=top_k,
            doc_content_chars_max=4000
        )
        docs = loader.load()
        return docs

    def _split(self, docs):
        split = self.text_splitter.split_documents([
            doc for doc in docs if not doc.page_content.startswith("Redirect")
        ])
        return split
    
    def _compress_docs(self, docs):
        compressed = []
        total_len = 0
        for doc in docs:
            if total_len + len(doc.page_content) > self.max_context:
                remaining = self.max_context - total_len
                compressed.append(doc.page_content[:remaining])
                break
            compressed.append(doc.page_content)
            total_len += len(doc.page_content)
        return "\n\n".join(compressed)
    
    def validate_fact(response):
        required_keys = [
            'sources', 'timestamps', 
            'confidence_score'
        ]
        return all(key in response.metadata for key in required_keys)