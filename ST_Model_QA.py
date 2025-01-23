import PyPDF2
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from re import search
import numpy as np

class PDFQAWithQdrant:
    def __init__(self, pdf_path, 
                 collection_name='qa_collection',
                 nlist = 100,
                 nprobe=5,
                 search_algo='HNSW',
                 ):
      
        self.pdf_path = pdf_path
        self.collection_name = collection_name
        self.text_chunks = []
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.embeddings = None


        self.nlist = nlist
        self.nprobe = nprobe
        self.search_algo = search_algo

        qdrant_url = "https://f2018bbf-fc1a-45dc-9f90-9f5c6f4834c2.europe-west3-0.gcp.cloud.qdrant.io:6333"
        qdrant_api_key = "NPnkE5dHAkIs05hmR8lvTT4NL6h0XxrPPdx5Q2XlsGobmiHexxd4Xg"

        self.qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )

        vector_params = VectorParams(size=384, distance=Distance.COSINE, on_disk=True)

        if not self.qdrant_client.collection_exists(self.collection_name):
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vector_params)
        

        self.tokenizer = None
        self.generator = None

        self._extract_text()
        self._split_text()
        self._load_generator()
        self._compute_and_store_embeddings()

    def _extract_text(self):
        reader = PyPDF2.PdfReader(self.pdf_path)
        self.full_text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            page_text = page.extract_text()
            if page_text:
                self.full_text += page_text + " "
        print("Page extraction complete.")

    def _split_text(self, chunk_size=500):
        words = self.full_text.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            self.text_chunks.append(chunk)
        print(f"Text split into {len(self.text_chunks)} chunks.")

    def _compute_and_store_embeddings(self):
        self.embeddings = self.embedding_model.encode(self.text_chunks, convert_to_tensor=True).cpu().numpy()
        print("Embeddings computed.")

        try:
          
            self.qdrant_client.get_collection(self.collection_name)
            collection_exists = True
        except Exception:
            collection_exists = False


      
        payloads = [{"text": chunk} for chunk in self.text_chunks]
        ids = list(range(len(self.text_chunks)))

        self.qdrant_client.upload_collection(
            collection_name=self.collection_name,
            vectors=self.embeddings.tolist(),  
            payload=payloads,
            ids=ids,
        )
        print(f"Stored {len(self.embeddings)} embeddings in Qdrant.")

    def _load_generator(self):
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.generator = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        print("Generator model loaded.")

    def _retrieve(self, question, top_k=3):
        question_embedding = self.embedding_model.encode(question, convert_to_tensor=True)

        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=question_embedding,
            limit=top_k,
        )

        retrieved_chunks = [hit.payload["text"] for hit in search_results]
        print(f"Retrieved {len(retrieved_chunks)} relevant chunks.")
        return retrieved_chunks

    def answer_question(self, question):
        retrieved_chunks = self._retrieve(question)
        context = " ".join(retrieved_chunks)

        input_text = f"question: {question} context: {context}"
        inputs = self.tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.generator.generate(inputs, max_length=150, num_beams=5, early_stopping=True)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return answer
    

