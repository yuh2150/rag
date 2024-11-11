from pydantic import BaseModel, Field

from src.rag.file_loader import Loader
from src.rag.vertorstore import VectorDB
from src.rag.offline_rag import Offine_RAG


class InputQA(BaseModel):
    question: str = Field(..., title="Question to ask the model ...")
class OutputQA(BaseModel):
    answer: str = Field(..., title="Answer to the given question ...")


def build_rag(llm, data_dir, data_type ):
    doc_loaded = Loader(file_type=data_type).load(data_dir, workers=2)
    retriver = VectorDB(documents=doc_loaded).get_retriever()
    rag_chain = Offine_RAG(llm).get_chain(retriver)
    
    return rag_chain