# SUTD T7 IR Project

### Information about dataset
[Patent file structure](patent_file_structure.md) <br />
[Data Analysis](data-analysis.md)


## Steps to run
1. Install [requirements.txt](requirements.txt)
```bash
pip install -r requirements.txt
```
2. Run [retriever evaluation](eval2.py)
```bash
python eval2.py
```
3. Run citation/generator evaluation in [llama_pinecone_citations](llama_pinecone_citations.ipynb)
4. Run hybrid BM25 + rerank search
```bash
python llama_pinecone_bm25_new.py
```
<br />
<br />
<br />
Evaluation dataset, original dataset and API keys will be provided upon request.