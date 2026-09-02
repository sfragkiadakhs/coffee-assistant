import requests
import time
import dlt
from gitsource import chunk_documents
import duckdb
from minsearch import Index
import os

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def build_index(chunks):
    index = Index(
        text_fields=['title', 'content'],
        keyword_fields=['doc_id']
    )
    index.fit(chunks)
    return index


def run_ingestion_pipeline():
    with open(DATA_DIR/"wiki_titles.txt") as f:
        titles = [line.strip() for line in f if line.strip()]

    pipeline = dlt.pipeline(
    pipeline_name="coffee",
    dataset_name="chunks",
    destination=dlt.destinations.duckdb(credentials=DATA_DIR/"coffee.duckdb")
    )

    info = pipeline.run(fetch_articles(titles),write_disposition="replace")


def load_index():
    if not os.path.exists(DATA_DIR/'coffee.duckdb'):
        run_ingestion_pipeline()

    with duckdb.connect(DATA_DIR/'coffee.duckdb') as con:
        chunks = con.execute('select doc_id, chunk_id, title, content from chunks.chunks').df().to_dict('records')

    return build_index(chunks)




def fetch_article(title, session):

    req_json = session.get(
        "https://en.wikipedia.org/w/api.php",
        params= {
            "action": "query", "prop": "extracts",
            "redirects": 1,  # some titles (e.g. "Arabica coffee") are redirects; follow them to get real content
            "explaintext":1, "format":"json", "titles":title,
        },
    ).json()
    time.sleep(0.1)  # be a polite API citizen

    return list(req_json['query']['pages'].values())[0]



@dlt.resource(table_name="chunks")
def fetch_articles(titles):
    headers = {"User-Agent": "CoffeeAssistant/0.1 (sfragkiadakhs@gmail.com)"}
    session = requests.Session()
    session.headers.update(headers)

    for title in titles:
        article = fetch_article(title, session)
        article_data = {
            "doc_id": article['pageid'],
            "title": article['title'],
            "text": article['extract']
        }
        
        # size=2000, step=1000: 1000-char overlap between consecutive chunks so a
        # passage spanning a chunk boundary still appears whole in at least one chunk
        chunks = chunk_documents([article_data], size=2000, step=1000, content_field_name="text")
        
        # number chunks per document (604727_1, 604727_2, ...) since chunk_documents
        # only gives us a "start" offset, not a stable id
        chunk_id = 0
        for chunk in chunks:
            doc_id = chunk['doc_id']
            chunk["chunk_id"] = f"{doc_id}_{chunk_id}"
            chunk_id += 1
            
            yield chunk