# SUTD T7 IR Project

Patent file structure

## Steps to run
1. Install requirements.txt
```bash
pip install -r requirements.txt
```
2. Run retriever evaluation
```bash
python eval2.py
```
3. Run citation/generator evaluation in [llama_pinecone_citations](llama_pinecone_citations.ipynb)
4. Run hybrid BM25 + rerank search
```bash
python llama_pinecone_bm25_new.py
```
