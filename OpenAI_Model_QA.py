import PyPDF2
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import openai
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class PDFQAWithQdrant:
    def __init__(self, pdf_path, 
                 collection_name='qa_collection',
                 nlist=100,
                 nprobe=5,
                 search_algo='HNSW',
                 openai_api_key=OPENAI_API_KEY):
      
        self.pdf_path = pdf_path
        self.collection_name = collection_name
        self.text_chunks = []
        self.openai_api_key = openai_api_key
        openai.api_key = self.openai_api_key  # Set OpenAI API key

        self.nlist = nlist
        self.nprobe = nprobe
        self.search_algo = search_algo

        qdrant_url = "https://f2018bbf-fc1a-45dc-9f90-9f5c6f4834c2.europe-west3-0.gcp.cloud.qdrant.io:6333"
        qdrant_api_key = "NPnkE5dHAkIs05hmR8lvTT4NL6h0XxrPPdx5Q2XlsGobmiHexxd4Xg"

        self.qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )

        vector_params = VectorParams(size=1536, distance=Distance.COSINE, on_disk=True)  # Adjust size for OpenAI embeddings

        if not self.qdrant_client.collection_exists(self.collection_name):
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vector_params)
        
        self._extract_text()
        self._split_text()
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
        embeddings = []
        for chunk in self.text_chunks:
            response = openai.embeddings.create(model="text-embedding-ada-002", input=chunk)
            embeddings.append(response.data[0].embedding)  # Accessing response data correctly
        
        print("Embeddings computed.")

        payloads = [{"text": chunk} for chunk in self.text_chunks]
        ids = list(range(len(self.text_chunks)))

        self.qdrant_client.upload_collection(
            collection_name=self.collection_name,
            vectors=embeddings,
            payload=payloads,
            ids=ids,
        )
        print(f"Stored {len(embeddings)} embeddings in Qdrant.")

    def _retrieve(self, question, top_k=3):
        question_embedding = openai.embeddings.create(model="text-embedding-ada-002", input=question).data[0].embedding

        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=question_embedding,
            limit=top_k,
        )

        retrieved_chunks = [hit.payload["text"] for hit in search_results]
        print(f"Retrieved {len(retrieved_chunks)} relevant chunks.")
        return retrieved_chunks

    # def answer_question(self, question):
    #     retrieved_chunks = self._retrieve(question)
    #     context = " ".join(retrieved_chunks)
        
    #     input_text = f"Answer the question based on the context.\n\nContext: {context}\n\nQuestion: {question}"
    #     response = openai.completions.create(
    #         model="gpt-3.5-turbo-instruct",
    #         prompt=input_text,
    #         temperature=0.5,

    #     )
        
    #     answer = response.choices[0].text.strip() 
    #     return answer


    def answer_question(self, question):
        retrieved_chunks = self._retrieve(question)
        context = " ".join(retrieved_chunks)
 
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Answer the question based on the context.\n\nContext: {context}\n\nQuestion: {question}"}
        ]

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=messages,
            temperature=0.5,
        )
        answer = response.choices[0].message.content.strip()
        return answer

    


    # def answer_question(self, question):
    #     retrieved_chunks = self._retrieve(question)
    #     context = " ".join(retrieved_chunks)
 
    #     messages = [
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": f"Answer the question based on the context.\n\nContext: {context}\n\nQuestion: {question}"}
    #     ]

    #     response = openai.chat.completions.create(
    #         model="gpt-4",
    #         messages=messages,
    #     )
    #     answer = response.choices[0].message.content.strip()
    #     return answer


    # def answer_question(self, question):
    #     retrieved_chunks = self._retrieve(question)
    #     context = " ".join(retrieved_chunks)
        
    #     messages = [
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": f"Answer the question based on the context.\n\nContext: {context}\n\nQuestion: {question}"}
    #     ]
        
    #     response = openai.chat.completions.create(
    #         model="gpt-4o",
    #         messages=messages,
    #     )
        
    #     answer = response.choices[0].message.content.strip()
    #     return answer

    # def answer_question(self, question):
    #     retrieved_chunks = self._retrieve(question)
    #     context = " ".join(retrieved_chunks)
        
    #     messages = [
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": f"Answer the question based on the context.\n\nContext: {context}\n\nQuestion: {question}"}
    #     ]
        
    #     response = openai.chat.completions.create(
    #         model="gpt-4-turbo",
    #         messages=messages,
    #         temperature=1,
    #     )
        
    #     answer = response.choices[0].message.content.strip()
    #     return answer

