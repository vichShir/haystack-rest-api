from typing import List

import pandas as pd
import logging

import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, APIRouter, UploadFile, File
from haystack.document_stores import BaseDocumentStore
from haystack.schema import Document
from haystack.nodes import PreProcessor

from rest_api.utils import get_app, get_pipelines
from rest_api.config import LOG_LEVEL
from rest_api.config import FILE_UPLOAD_PATH
from rest_api.schema import FilterRequest


logging.getLogger("haystack").setLevel(LOG_LEVEL)
logger = logging.getLogger("haystack")


router = APIRouter()
app: FastAPI = get_app()
document_store: BaseDocumentStore = get_pipelines().get("document_store", None)


@router.post("/documents/get_by_filters", response_model=List[Document], response_model_exclude_none=True)
def get_documents(filters: FilterRequest):
    """
    This endpoint allows you to retrieve documents contained in your document store.
    You can filter the documents to retrieve by metadata (like the document's name),
    or provide an empty JSON object to clear the document store.

    Example of filters:
    `'{"filters": {{"name": ["some", "more"], "category": ["only_one"]}}'`

    To get all documents you should provide an empty dict, like:
    `'{"filters": {}}'`
    """
    docs = [doc.to_dict() for doc in document_store.get_all_documents(filters=filters.filters)]
    for doc in docs:
        doc["embedding"] = None
    return docs


@router.post("/documents/delete_by_filters", response_model=bool)
def delete_documents(filters: FilterRequest):
    """
    This endpoint allows you to delete documents contained in your document store.
    You can filter the documents to delete by metadata (like the document's name),
    or provide an empty JSON object to clear the document store.

    Example of filters:
    `'{"filters": {{"name": ["some", "more"], "category": ["only_one"]}}'`

    To get all documents you should provide an empty dict, like:
    `'{"filters": {}}'`
    """
    document_store.delete_documents(filters=filters.filters)
    return True


@router.post("/documents/insert_csv", response_model=str)
def write_documents(file: UploadFile = File(...)):

    # Clear document store
    document_store.delete_documents()

    # Load file
    file_path: str = ''
    try:
        file_path = Path(FILE_UPLOAD_PATH) / f"{uuid.uuid4().hex}_{file.filename}"
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()

    # Open csv
    df = pd.read_csv(file_path)
    df.rename(columns={
        'DESCRIÇÃO LONGA': 'descricao',
        'TAGS': 'tags',
        'CATEGORIA': 'categoria',
        'NOME DA STARTUP': 'title'
    }, inplace=True)
    df['text'] = df['descricao'] + ' ' + df['tags'] + ' ' + df['categoria'] + ' | Nome da startup: ' + df['title']
    df = df[['title', 'text']]
    df.fillna(value="", inplace=True)

    titles = list(df["title"].values)
    texts = list(df["text"].values)

    # Create to haystack document format
    documents = []
    for title, text in zip(titles, texts):
        documents.append(Document(content=text, meta={"name": title or ""}))

    # Preprocessing
    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=False,
        split_by="word",
        split_length=100,
        split_respect_sentence_boundary=True,
        language='pt'
    )
    docs_default = preprocessor.process(documents)

    document_store.write_documents(docs_default)
    return f"Successfully uploaded {file.filename}"