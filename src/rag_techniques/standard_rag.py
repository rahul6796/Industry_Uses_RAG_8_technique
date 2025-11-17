

from typing import List, Dict
from pyprojroot import here
from langchain.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from load_config import APPConfig
import chromadb
from langchain.schema import Document
APP_CONFIG = APPConfig.load()


class StandardRAG:


    def __init__(self):

        self.embedding =  ## define the embedding model 
        self.llm = # define the llm model
        self.logs = []
        self.retrievers = {}


    def _setup_retrievers(self):
        """
        setup retriever for all the datasets.
        """
        datasets = ["tech_docs", "faq_data",
                    "news_articles"]


        for dataset in datasets:

            try:
                client = chromadb.PersistentClient(
                    path=str(here(APP_CONFIG.chroma_db_path)))
                collection = chroma_client.get_collection(dataset)


                class CustomRetriever:

                    def __init__(self, collection, embedding, logger):
                        self.collection = collection
                        self.embedding = embedding
                        self.logger = logger

                    def get_relavent_documents(self, query: str, k: int=5):
                        try:
                            query_embedding = self.embedding.embed_query(
                                query
                            )

                            resutl = self.collection.query(
                                query_embeddings = [query_embedding],
                                n_result = k,
                                include = ['documents', 'metadatas']
                            )

                            documents = []
                            if results['documents'] and results['documents'][0]:

                                for i, doc_content in enumerate(results['documents'][0]):
                                    metadata = result['metadatas'][0][i] if result['metadata'] else {}
                                    documents.append(
                                            Document(page_content=doc_content, metadata=metadata))

                            self.logger(
                                f"Retrieved {len(documents)} documents using similarity search")
                        except Exception as e:
                            print(f'error get relavent documents {e}')
                            return []
                
                self.retrievers[dataset] = CustomRetriever(
                    collection, self.embeddings, self._logs
                )

            except Exception as e:
                self._log(f"Setup error for {dataset}: {str(e)}")


    def _log(self, message: str):
        """
        simple logging
        """
        log_entry = f"STANDARD RAG: {message}"
        self.logs.append(log_entry)

    
        if any(step in message for step in ["Step 1:", "Step 2:"]):
            self.logs.append("     |")
            self.logs.append("     |")
            self.logs.append("     V")



    def process_query(self, query: str, dataset: str)-> Tuple[str, List(str)]:
        self.logs = []
        self._log("Starting Standard RAG pipeline")
    

        try:

            if dataset not in self.retrievers:
                return f"Dataset {dataset} not aviable". self.logs

            retriever = self.retrievers[dataset]



            # RAG prompt:
            template =  """Answer the question based on the context:

            {context}

            Question: {question}

            Answer:"""

            prompt = ChatPromptTemplate.from_template(template)
            self._log("Using simple retrieve-then-generate approach")


            # Document formatter
            def format_docs(docs):
                if not docs:
                    self._log("No documents retrieved")
                    return "No relevant documents found."

                self._log(f"Formatting {len(docs)} documents into context")
                return "\n\n".join(doc.page_content for doc in docs if doc.page_content)

            # RAG Chain

            def retrieve_and_format(query_input):
                self._log("Setp 1: Retrieving the relavent documents")
                docs = retriever.get_relevant_documents(
                    query_input, k=APP_CONFIG.standard_rag.top_k)
                formatted = format_docs(docs)
                self._log("Step 2: Generating response with retrieved context")
                return formatted
            

            rag_chain = (
                "context": retrieve_and_format, "question": RunnablePassthrough()
                | prompt
                |self.llm
                | StrOutputParser()
            )

            response = rag_chain.invoke(query)
            self._log(
                "Standard RAG completed: Single retrieval → Direct generation")

            return response, self.logs

        except Exception as e:
            error_msg = f"RAG failed: {str(e)}"
            self._log(error_msg)
            return f"Error processing request: {str(e)}", self.logs

        


        








