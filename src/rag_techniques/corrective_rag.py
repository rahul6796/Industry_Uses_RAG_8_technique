from typing import List, Tuple
from pyprojroot import here
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
from load_config import APPConfig
import chromadb



APP_CONFIG = APPConfig.load()

class CorrectiveRAG:

    def __init__(self):
        self.embeddings = # implement the embedding load model code.
        self.llm = # implement the llm load model code.
        self.logs= []
        self.retrievers = {}

        self._setup_retrievers()
        self._setup_graders() ## used to correct the answer or documents before generate the answer by the llm.


    def _setup_retrievers(self):
        """Setup retrievers for all datasets"""
        datasets = ["tech_docs", "faq_data", "news_articles"]

        for dataset in datasets:
            try:
                
                # get the collection from the dataset.
                chroma_client = chromadb.PersistentClient(
                    path=str(here(APP_CONFIG.chroma_db_path)))
                collection = chroma_client.get_collection(dataset)


                # Define the Custom Retriever.
                class CustomRetriever:

                    def __init__(self, collection, embedding, logger):
                        self.collection = collection
                        self.embedding = embedding
                        self.logger = logger


                    def get_relavent_documents(self, query: str, k: int=3):

                        try:
                            query_embedding = self.embeddings.embed_query(
                                query)

                            results = self.collection.query(
                                query_embeddings=[query_embedding],
                                n_results=k,
                                include=['documents', 'metadatas']
                            )

                            documents = []
                            if results['documents'] and results['documents'][0]:
                                for i, doc_content in enumerate(results['documents'][0]):
                                    metadata = results['metadatas'][0][i] if results['metadatas'][0] else {
                                    }
                                    documents.append(
                                        Document(page_content=doc_content, metadata=metadata))
                            else:
                                pass
                            return  documents
                        except Exception as e:
                            self.logger(f"Retrieval error: {str(e)}")
                            return []

                # getting retrievers.
                self.retrievers[dataset] = CustomRetriever(
                    collection, self.embeddings, self._log)

            except Exception as e:
                self._log(f"Setup error for {dataset}: {str(e)}")
       

    def _setup_graders(self):
        
        """
        Setup document grading and query rewriting models
        """

         # Document relevance grader
         class GradeDocuments(BaseModel):
            """Binary score for relevance check on retrieved documents."""

            binary_score: str = Field(
                description="Documents are relevant to the question, 'yes' or 'no'"
            )

        self.doc_grader_llm = self.llm.with_structured_output(GradeDocuments)

            # Document grading prompt
            grade_system = """You are a grader assessing relevance of retrieved documents to a user question.
    
                        If the document contains keywords or semantic meaning related to the question, grade it as relevant.
                        The goal is to filter out erroneous retrievals that don't help answer the question.
                        Give a binary score 'yes' or 'no'."""


        self.grade_prompt = ChatPromptTemplate.from_messages([
        ("system", grade_system),
        ("human",
            "Retrieved document: \n\n {document} \n\n User question: {question}")])

        self.doc_grader = self.grade_prompt | self.doc_grader_llm


        # Query rewriter for better retrieval
        rewrite_system = """You are a question re-writer that converts an input question to a better version optimized for vectorstore retrieval.
        
                            Look at the input and try to reason about the underlying semantic intent/meaning.
                            Improve the question by:
                            - Making it more specific and clear
                            - Adding relevant keywords
                            - Maintaining the original intent"""

        self.rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", rewrite_system),
            ("human",
             "Here is the initial question: \n\n {question} \n Formulate an improved question.")
        ])
        self.query_rewriter = self.rewrite_prompt | self.llm | StrOutputParser()

    
    def _log(self):
        """Enhanced logging with visual separators"""
        log_entry = f"CORRECTIVE RAG: {message}"
        self.logs.append(log_entry)

        # Add visual separator after key steps
        if any(step in message for step in ["Step", "Decision:", "Correction:"]):
            self.logs.append("     |")
            self.logs.append("     |")
            self.logs.append("     V")

    

    def _web_search(self, query: str) -> str:
        """Optimized web search with token limits"""

        try:
            self._log(
                "Web Search: Using OpenAI's web search tool (minimal tokens)")

            client = OpenAI()

            # Create a more focused search query
            # Limit query length!!!
            focused_query = f"Brief summary: {query[:50]}"

            response = client.responses.create(
                model=APP_CONFIG.corrective_rag.web_search_model,
                tools=[{
                    "type": "web_search_preview",
                    "search_context_size": "low"
                }],
                input=f"Give a concise 2-sentence answer for: {focused_query}"
            )

            web_content = response.output_text

            if len(web_content) > 2000:
                web_content = web_content[:2000] + "..."
                self._log(
                    "Web Search: Truncated results to 2000 chars for efficiency")

            self._log(
                f"Web Search: Retrieved {len(web_content)} chars from web")

            return web_content

        except Exception as e:
            self._log(f"Web Search: Failed - {str(e)}")
            return f"Current web information unavailable for: {query}"


    def _grade_documents(self, query: str, documents: List) -> Tuple[List, bool]:

        """Grade document relevance and determine if web search is needed"""

        if not documents:
            self._log("Document Grading: No documents to grade")
            return [], True

        self._log(f"Document Grading: Evaluating {len(documents)} documents")

        relevant_docs = []
        need_web_search = False


        for i doc in enumerate(documents):

            try:
                score = self.doc_grader.invoke({
                    "question": query,
                    "document": doc.page_content
                })

                if score.binary_score.lower() == "yes":
                    relevant_docs.append(doc)
                    self._log(f"Document {i+1}: RELEVANT - keeping")
                else:
                    need_web_search = True
                    self._log(
                        f"Document {i+1}: NOT RELEVANT - will need web search")

            except Exception as e:
                self._log(f"Document {i+1}: Grading failed - {str(e)}")
                need_web_search = True

        relevance_ratio = len(relevant_docs) / len(documents)
        self._log(
            f"Document Grading: {len(relevant_docs)}/{len(documents)} relevant ({relevance_ratio:.1%})")
        

        # Decision logic: need web search if too few relevant docs
        if len(relevant_docs) == 0:
            self._log(
                "Decision: NO relevant documents found - web search required")
            need_web_search = True
        elif relevance_ratio < APP_CONFIG.corrective_rag.relevance_ratio:
            self._log("Decision: LOW relevance ratio - web search recommended")
            need_web_search = True
        else:
            self._log(
                "Decision: SUFFICIENT relevant documents - no web search needed")
            need_web_search = False

        return relevant_docs, need_web_search













    

    def process_query(self):
        pass
