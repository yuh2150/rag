from typing import Union, List, Literal
import glob
from tqdm import tqdm
import multiprocessing
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def remove_non_utf8_characters(text: str) -> str:
    return ''.join(char for char in text if ord(char) < 128)

def load_pdf(pdf_file: str) -> List:
    docs = PyPDFLoader(pdf_file, extract_images=True).load()
    for doc in docs:
        doc.page_content = remove_non_utf8_characters(doc.page_content)
    return docs

def get_num_cpu() -> int:
    return multiprocessing.cpu_count()

class BaseLoader:
    def __init__(self) -> None:
        self.num_processes = get_num_cpu()

    def __call__(self, files: List[str], **kwargs):
        pass

class PDFLoader(BaseLoader):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, pdf_files: List[str], **kwargs):
        num_processes = min(self.num_processes, kwargs["workers"])

        with multiprocessing.Pool(processes=num_processes) as pool:
            doc_loaded = []
            total_files = len(pdf_files)            
            with tqdm(total=total_files, desc="Loading PDFs", unit="file") as pbar:
                for result in pool.imap_unordered(load_pdf, pdf_files):
                    doc_loaded.extend(result)
                    pbar.update(1)
        
        return doc_loaded

class TextSplitter:
    def __init__(
        self,
        separators: List[str] = ['\n\n', '\n', '', ' '],
        chunk_size: int = 300,
        chunk_overlap: int = 8
    ) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def __call__(self, documents):
        return self.splitter.split_documents(documents)
    
class Loader:
    def __init__(
        self,
        file_type: Literal["pdf"],
        split_kwargs: dict = {
            "chunk_size": 300,
            "chunk_overlap": 8
        }
    ) -> None:
        assert file_type in ["pdf"], "file_type must be 'pdf'"
        self.file_type = file_type
        if file_type == "pdf":
            self.doc_loader = PDFLoader()
        else:
            raise ValueError("file_type must be 'pdf'")
        
        self.doc_splitter = TextSplitter(**split_kwargs)

    def load(self, pdf_files: Union[str, List[str]], workers: int = 1):
        if isinstance(pdf_files, str):
            pdf_files = [pdf_files]
        
        doc_loaded = self.doc_loader(pdf_files, workers=workers)
        doc_split = self.doc_splitter(doc_loaded)
        return doc_split

    def load_dir(self, dir_path: str, workers: int = 1):
        files = glob.glob(f"{dir_path}/*.pdf")
        if self.file_type == "pdf":
            assert len(files) > 0, f"No {self.file_type} files found in {dir_path}"
        else:
            raise ValueError("file_type must be 'pdf'")
        
        return self.load(files, workers=workers)