import os
import time
from typing import List
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from src.utils import load_keys
from pinecone import Pinecone
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone as PineconeLang
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_openai import ChatOpenAI
from pinecone import ServerlessSpec
from langchain.schema import SystemMessage
from langchain_core.embeddings import Embeddings
# Load environment variables
os.environ["OPENAI_API_KEY"] = load_keys()["openai"]
os.environ["DEEPGRAM_API_KEY"] = load_keys()["deepgram"]
os.environ["PINECONE_API_KEY"] = load_keys()["pinecone1"]

from dotenv import load_dotenv
load_dotenv()

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = "us-east-1"  # Adjust to your Pinecone environment
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

# Initialize Pinecone


# Load the Alibaba model for embedding generation
model_path = 'Alibaba-NLP/gte-base-en-v1.5'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
  #.to('cuda') Move model to GPU


class EmbeddingGenerator(Embeddings):
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Tokenize the input texts and move input tensors to GPU
        batch_dict = self.tokenizer(texts, max_length=8192, padding=True, truncation=True, return_tensors='pt')#.to('cuda')

        # Pass inputs through the model
        with torch.no_grad():  # Disable gradient calculation for inference
            outputs = self.model(**batch_dict)

        # Check for expected output format
        if not hasattr(outputs, 'last_hidden_state'):
            raise ValueError("Model output does not contain 'last_hidden_state'.")

        # Extract embeddings and normalize
        embeddings = outputs.last_hidden_state[:, 0]  # Take the [CLS] token representation
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)

        return normalized_embeddings.detach().cpu().numpy().tolist()  # Convert to list of lists

    def embed_query(self, text: str) -> List[float]:
        # Generate embeddings for the query (single string)
        return self.embed_documents([text])[0]  # Call embed_documents and return the first element




class LLMLangchain:

    def __init__(self) -> None:
        self.chat = ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], model='gpt-4o-mini')
        self.embedding = EmbeddingGenerator(tokenizer, model)
        self.pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        self.spec = ServerlessSpec(
            cloud="aws", region="us-east-1"
        )
        self.index_name = "issf"
        self.index = self.init_pinecone()
        # self.messages = [
        #     SystemMessage(content="You are a helpful answer provider for Duck Creek Forms development and you will be asked the questions related to Duck Creek forms functionality to get the information."),
        # ]

        self.messages = [
                    SystemMessage(content= """You are a knowledgeable assistant trained to provide detailed and accurate information from the "Shooting Sports Handbook." 
                                The handbook covers various rules, safety guidelines, equipment specifications, competition procedures, and numerical data related to shooting sports. When answering questions, ensure that:

        You cite specific rule numbers, measurements, and technical specifications whenever applicable.
        You provide clear and concise answers, avoiding ambiguity, especially for questions involving numerical or measurable data.
        You explain shooting sport regulations accurately, including safety protocols, scoring methods, target types, and equipment standards.
        If the question involves measurements (distances, scoring zones, or equipment dimensions), your response must be precise and derived from the handbook's standards.
                                ou are expected to answer questions that might include:

        The specifics of competition procedures.
        The required safety protocols during events.
        Target and scoring standards for various events (e.g., Rifle, Pistol, Shotgun).
        Clothing and equipment standards for athletes.
        Numerical data regarding target sizes, distances, and timing rules.
        Always ensure the responses reflect the strict regulations and numerical accuracy provided in the handbook.
                                You can take your time to generate and provide relevant and accurate answers.       
                                """)]

    def init_pinecone(self):
        # Connect to Pinecone index
        self.index = self.pc.Index(self.index_name)
        time.sleep(1)
        return self.index

    # def generate_insert_embeddings_(self, docs):
    #     for j in range(len(docs)):
    #         embeddings_metadata = [{"metadata": ""}]
    #         ids = [str(j)]
            
    #         # Generate embeddings using the Alibaba model
    #         embeds = self.embedding.embed_documents([docs[j].page_content])
    #         embeddings_metadata[0]["metadata"] = str(docs[j].page_content)

    #         # Move embeddings to CPU and convert to list of floats for Pinecone insertion
    #         embeds = embeds.detach().cpu().numpy().tolist()

    #         # Upsert into Pinecone
    #         self.index.upsert(vectors=zip(ids, embeds, embeddings_metadata))


    def generate_insert_embeddings_(self, docs):

        for j in range(len(docs)):
            embeddings = [{"metadata": ""}]

            ids = [str(j)]
            # Generate embeddings and ensure it's a flat list of floats
            embeds = self.embedding.embed_documents(docs[j].page_content)
            embeddings[0]["metadata"]= str(docs[j].page_content)


            self.index.upsert(vectors=zip(ids, embeds, embeddings))


    def retrieve_query(self, index, query, k=2):
        matching_results = index.similarity_search(query, k=k)
        return matching_results

    def read_doc(self, directory):
        file_loader = PyPDFDirectoryLoader(directory)
        documents = file_loader.load()
        return documents

    def chunk_data(self, docs, chunk_size=800, chunk_overlap=50):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.doc = self.text_splitter.split_documents(docs)
        return self.doc


    def augment_prompt(self, query: str):

        vectorstore = PineconeLang(
            self.index, self.embedding.embed_query, "metadata"
        )

        results = vectorstore.similarity_search(query=query, k=5)  # Pass the embedding correctly


        # Get the text from the results
        source_knowledge = "\n".join([x.page_content for x in results])

        # Print the results for debugging
        for i, x in enumerate(results):
            print(f"Chunk {i + 1}: ", x.page_content)

        # Construct the augmented prompt
        augmented_prompt = f"""Using the contexts below, answer the query.

        Contexts:
        {source_knowledge}

        Query: {query}"""
        return augmented_prompt



# # Main execution to generate embeddings and insert into Pinecone
# if __name__ == "__main__":
#     llm = LLMLangchain()

#     ######################################################################
#     ## To Generate new embeddings and insert into Pinecone
#     print("Embedding Started")
#     doc = llm.read_doc("Database/")  # Update "Database/" with your actual document directory
#     documents = llm.chunk_data(docs=doc)
#     print(f"Total chunks: {len(documents)}")
#     embeddings = llm.generate_insert_embeddings_(documents)
#     print("Embedding Completed and inserted into Pinecone")
#     txt="tell me about printformrq"
#     # embed = generate_embeddings([txt])
#     # print(embed)
#     prompt=llm.augment_prompt(query=txt)
