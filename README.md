# Shooting-Sports-Mastermind

This project focuses on building a Retrieval-Augmented Generation (RAG) chatbot designed for the highly regulated and precise world of shooting sports. The chatbot assists shooters, juries, and other stakeholders by answering domain-specific questions related to rules and regulations governing weapon settings, safety, measurements, and more. 
<br>
The chatbot leverages LangChain, Pinecone Vector DB, and OpenAI models to deliver accurate and fast responses based on a robust knowledge base.

## Features

**Text Embeddings:** Uses gte-base embeddings for efficient document retrieval. <br>
**Fast Query Handling:** Powered by gpt-4.o mini to generate prompt and contextually accurate responses. <br>
**Seamless CI/CD Pipeline:** Fully automated deployment using GitHub Actions with hosting on Heroku. <br>
<br>

## Data Sources
The knowledge base is built using shooting-related PDFs collected from various online and offline sources. The PDF content includes, but is not limited to:
 <br>
- Weapon setup guidelines (Rifle/Pistol/Shotgun)<br>
- Safety protocols <br>
- Measurement rules <br>
- Anti-doping Rules
- Disqualification rules <br>


## Installation & Setup
**1. Clone the Repository:** <br>
     ``` 
     git clone https://github.com/pragatinaikare/RAG_BASED_CHATBOT.git
     ```<br>

**2. Install Dependencies:**

```bash
pip install -r requirements.txt
```

**3. Set Up API Keys:** 
You will need to create accounts with Pinecone and OpenAI to get your API keys. Add these keys in keys.txt file inside Data Folder


## References
1] [Searching for Best Practices in Retrieval-Augmented Generation](https://arxiv.org/abs/2407.01219)<br>
2] [Context Embeddings for Efficient Answer Generation in RAG](https://arxiv.org/abs/2407.09252)<br>
3] [RAGAS: Automated Evaluation of Retrieval-Augmented Generation](https://arxiv.org/abs/2309.15217)<br>
