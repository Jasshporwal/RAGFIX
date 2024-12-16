from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
from fastapi import HTTPException
from typing import List, Dict
import openai
import logging
from config.settings import settings
from .pinecone_manager import pinecone_manager
from langchain_openai import OpenAIEmbeddings
import hashlib
import nltk
from nltk import pos_tag, word_tokenize
from nltk.tree import Tree
from collections import defaultdict
import os 
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

logger = logging.getLogger(__name__)

class RAGFix:
    def __init__(self):
        self.embedding_dimension = settings.EMBEDDING_DIMENSION
        self.max_tokens = settings.MAX_TOKENS
        self.temperature = settings.TEMPERATURE
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.model = "text-embedding-ada-002"
        self.embed = OpenAIEmbeddings(model=self.model, openai_api_key=self.openai_api_key)
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        #  Initialize Groq client
        self.groq_client = Groq(api_key=self.groq_api_key)

    def _generate_chunk_id(self, content: str, source: str, chunk_index: int) -> str:
        """
        Generate a unique ID for a chunk using its content, source file name, and chunk index.
        """
        unique_string = f"{source}_{chunk_index}_{content}"
        return hashlib.md5(unique_string.encode()).hexdigest()
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for the given text.
        """
        try:
            logger.debug(f"Input text for embedding: '{text}'")
            if not text or not text.strip():
                raise ValueError("Input text is empty or invalid.")
            
            embeddings = self.embed.embed_documents([text])
            
            # Flatten the embedding if it's nested
            if isinstance(embeddings, list) and isinstance(embeddings[0], list):
                return embeddings[0]
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

    def process_folder(self, folder_path: str, overlap: int = 200, chunk_size: int = 1024):
        """
        Process a folder of `.txt` files: split, embed, and store chunks in Pinecone.
        Skip chunks that have already been processed.
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".txt")]
        if not files:
            raise HTTPException(status_code=404, detail="No text files found in the folder.")

        logger.info(f"Found {len(files)} text files in the folder: {folder_path}")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        
        for file_path in tqdm(files, desc="Processing files"):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                chunks = text_splitter.split_text(content)
                file_name = os.path.basename(file_path)
                
                for idx, chunk in enumerate(chunks):
                    try:
                        # Generate a unique ID for the chunk
                        chunk_id = self._generate_chunk_id(chunk, file_name, idx)
                        
                        # Skip if the chunk already exists in Pinecone
                        if pinecone_manager.vector_exists(chunk_id):
                            logger.info(f"Chunk {chunk_id} already processed. Skipping.")
                            continue
                        
                        # Generate embedding and upsert into Pinecone
                        embedding = self.generate_embedding(chunk)
                        pinecone_manager.upsert_embedding(
                            vector_id=chunk_id,
                            embedding=embedding,
                            metadata={
                                "source": file_name,
                                "chunk_index": idx,
                                "content": chunk,
                            }
                        )
                    except Exception as e:
                        logger.error(f"Error processing chunk {idx} of file {file_name}: {str(e)}")
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")

    def extract_noun_phrases(self, text: str) -> List[str]:
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        grammar = "NP: {<DT>?<JJ>*<NN>+}"
        cp = nltk.RegexpParser(grammar)
        tree = cp.parse(tagged)
        noun_phrases = [
            " ".join(word for word, tag in subtree.leaves())
            for subtree in tree.subtrees()
            if isinstance(subtree, Tree) and subtree.label() == "NP"
        ]
        return noun_phrases
    
    def detect_conflicts(self, facts: List[Dict]) -> List[Dict]:
        """
        Detect and resolve conflicts in retrieved facts.
        """
        logger.debug(f"Detecting conflicts in facts: {facts}")
        noun_phrase_to_facts = defaultdict(list)
        
        for fact in facts:
            noun_phrases = self.extract_noun_phrases(fact["content"])
            for np in noun_phrases:
                noun_phrase_to_facts[np].append(fact)

        consistent_facts = []
        for np, fact_group in noun_phrase_to_facts.items():
            if len(fact_group) == 1:
                consistent_facts.extend(fact_group)
            else:
                logger.warning(f"Conflict detected for noun phrase '{np}':")
                for fact in fact_group:
                    logger.warning(f"Conflicting fact: {fact['content']} (Source: {fact['source']})")
        
        logger.debug(f"Consistent facts after resolving conflicts: {consistent_facts}")
        return consistent_facts


    def retrieve_facts(self, embedding: list, top_k: int = 5) -> list:
        """
        Retrieve top matching facts for a given embedding from Pinecone.
        """
        try:
            if not pinecone_manager.index:
                raise HTTPException(status_code=500, detail="Pinecone index is not initialized.")

            results = pinecone_manager.query_embeddings(query_embedding=embedding, top_k=top_k)
            if not results:
                raise HTTPException(status_code=404, detail="No matches found in Pinecone.")

            return results
        except Exception as e:
            logger.error(f"Error retrieving facts from Pinecone: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Fact retrieval failed: {str(e)}")
    
    def verify_facts(self, facts: List[Dict]) -> List[Dict]:
        """
        Verify and filter facts, resolving any conflicts.
        """
        logger.debug(f"Verifying facts: {facts}")
        filtered_facts = [fact for fact in facts if "source" in fact]
        consistent_facts = self.detect_conflicts(filtered_facts)
        logger.debug(f"Consistent facts after conflict detection: {consistent_facts}")
        return consistent_facts
    
    ### Using OpenAI API ###
    # def generate_response(self, query: str, facts: list) -> str:
    #     """
    #     Generate a response based on the retrieved facts.
    #     """
    #     if not facts:
    #         return "I'm sorry, I couldn't find enough consistent information to answer your query."

    #     max_facts = 5
    #     limited_facts = facts[:max_facts]

    #     facts_context = "\n".join(
    #         [f"Fact: {fact['content']} (Source: {fact['source']})" for fact in limited_facts]
    #     )

    #     prompt = f"""Using the verified facts below, respond accurately to the query without speculating:

    #     Facts:
    #     {facts_context}

    #     Query: {query}

    #     Response:"""

    #     try:
    #         response = openai.chat.completions.create(
    #             model=settings.MODEL_NAME,
    #             messages=[
    #                 {
    #                     "role": "system",
    #                     "content": "You are a fact-based assistant. Only use the provided facts to answer questions. If you're unsure, say so explicitly.",
    #                 },
    #                 {"role": "user", "content": prompt},
    #             ],
    #             temperature=self.temperature,
    #             max_tokens=self.max_tokens,
    #         )

    #         if not response.choices:
    #             raise ValueError("OpenAI API returned an empty response.")

    #         return response.choices[0].message.content.strip()
    #     except Exception as e:
    #         logger.error(f"Error generating response: {str(e)}")
    #         raise HTTPException(status_code=500, detail=f"Response generation failed: {str(e)}")
        
### GROQ implementation FOR LLAMA70B, LLMA8B, MIXTRAL JUST CHANGE THE MODEL NAME###

    def generate_response(self, query: str, facts: list) -> str:
        """
        Generate a response based on the retrieved facts using Groq's Llama 3.
        """
        if not facts:
            return "I'm sorry, I couldn't find enough consistent information to answer your query."

        max_facts = 5
        limited_facts = facts[:max_facts]

        facts_context = "\n".join(
            [f"Fact: {fact['content']} (Source: {fact['source']})" for fact in limited_facts]
        )

        prompt = f"""Using the verified facts below, respond accurately to the query without speculating:

        Facts:
        {facts_context}

        Query: {query}

        Response:"""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama3-groq-8b-8192-tool-use-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a fact-based assistant. Only use the provided facts to answer questions. If you're unsure, say so explicitly.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            if not response.choices:
                raise ValueError("Groq API returned an empty response.")

            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Response generation failed: {str(e)}")

ragfix = RAGFix()