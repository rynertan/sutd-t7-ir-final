from llama_index.core import (
    Document,
    Settings,
    VectorStoreIndex,
    StorageContext,
    QueryBundle,
)
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import NodeWithScore
from pinecone import Pinecone
from pathlib import Path
import json
from tqdm import tqdm
import logging
import os
from typing import List, Optional, Iterator, Dict, Any, Set
from dotenv import load_dotenv, find_dotenv
import pickle
from datetime import datetime
import time
from tenacity import retry, wait_exponential, stop_after_attempt
from collections import defaultdict
import pandas as pd
from dataclasses import dataclass
import sys


@dataclass
class RetrievalConfig:
    """Configuration for retrieval methods."""

    method: str = "vector"  # "vector", "bm25", or "hybrid"
    top_k: int = 5
    use_reranking: bool = False
    reranking_model: str = "BAAI/bge-reranker-base"
    rerank_top_n: int = 4


class HybridRetriever(BaseRetriever):
    """Custom hybrid retriever that combines vector and BM25 search."""

    def __init__(self, vector_retriever, bm25_retriever: BM25Retriever, top_k: int = 5):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.top_k = top_k
        super().__init__()

    def _retrieve(self, query: str, **kwargs) -> List[NodeWithScore]:
        """Retrieve nodes using both vector and BM25 methods."""
        bm25_nodes = self.bm25_retriever.retrieve(query)
        vector_nodes = self.vector_retriever.retrieve(query)

        # Combine results while avoiding duplicates
        all_nodes = []
        node_ids = set()
        for node in bm25_nodes + vector_nodes:
            if node.node.node_id not in node_ids:
                all_nodes.append(node)
                node_ids.add(node.node.node_id)

        return all_nodes[: self.top_k]


class PatentIndexBuilder:
    def __init__(
        self,
        data_dir: str,
        embed_model: str = "text-embedding-3-small",
        openai_api_key: Optional[str] = None,
        pinecone_api_key: Optional[str] = None,
        pinecone_region: Optional[str] = None,
        batch_size: int = 100,
        checkpoint_dir: Optional[str] = None,
    ):
        """Initialize the patent index builder with vector and BM25 capabilities."""
        load_dotenv(find_dotenv())

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints")
        )

        # API setup
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        self.pinecone_region = pinecone_region or os.getenv("PINECONE_REGION")

        # Validate
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {data_dir} does not exist")
        if not all([self.openai_api_key, self.pinecone_api_key, self.pinecone_region]):
            raise ValueError("Missing required API keys or region")

        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.pinecone_api_key)

        # Configure embedding
        self.embed_model = OpenAIEmbedding(
            model_name=embed_model, api_key=self.openai_api_key, dimensions=1536
        )
        Settings.embed_model = self.embed_model
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50

        # Initialize retrievers and reranker as None
        self.vector_index = None
        self.bm25_retriever = None
        self.hybrid_retriever = None
        self.reranker = None

        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def get_checkpoint_paths(self) -> tuple[Path, Path]:
        """Get paths for both processing and embedding checkpoints."""
        return (
            self.checkpoint_dir / "processing_checkpoint.pkl",
            self.checkpoint_dir / "embedding_checkpoint.pkl",
        )

    def load_checkpoints(self) -> tuple[set, set]:
        """Load both processing and embedding checkpoints."""
        proc_path, embed_path = self.get_checkpoint_paths()
        processed_ids = set()
        embedded_ids = set()

        if proc_path.exists():
            with open(proc_path, "rb") as f:
                try:
                    checkpoint_data = pickle.load(f)
                    processed_ids = checkpoint_data.get("document_ids", set())
                    self.logger.info(
                        f"Loaded processing checkpoint: {len(processed_ids)} documents processed"
                    )
                except Exception as e:
                    self.logger.warning(f"Error loading processing checkpoint: {e}")
                    processed_ids = set()

        if embed_path.exists():
            with open(embed_path, "rb") as f:
                try:
                    checkpoint_data = pickle.load(f)
                    embedded_ids = checkpoint_data.get("document_ids", set())
                    self.logger.info(
                        f"Loaded embedding checkpoint: {len(embedded_ids)} documents embedded"
                    )
                except Exception as e:
                    self.logger.warning(f"Error loading embedding checkpoint: {e}")
                    embedded_ids = set()

        return processed_ids, embedded_ids

    def save_checkpoint(self, checkpoint_type: str, ids: set):
        """Save checkpoint for either processing or embedding."""
        proc_path, embed_path = self.get_checkpoint_paths()
        path = proc_path if checkpoint_type == "processing" else embed_path

        checkpoint_data = {"document_ids": ids, "timestamp": datetime.now().isoformat()}

        with open(path, "wb") as f:
            pickle.dump(checkpoint_data, f)
        self.logger.info(f"Saved {checkpoint_type} checkpoint: {len(ids)} documents")

    def process_patent(self, md_file: Path, metadata_dir: Path) -> Optional[Document]:
        """Process a single patent file and return a Document object."""
        try:
            # Read content
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if not content:
                self.logger.warning(f"Empty content in {md_file}")
                return None

            # Read metadata
            patent_id = md_file.stem
            metadata = {"patent_id": patent_id}
            metadata_file = metadata_dir / f"{patent_id}.json"

            if metadata_file.exists():
                with open(metadata_file, "r", encoding="utf-8") as f:
                    full_metadata = json.load(f)
                    metadata.update(
                        {
                            "patent_number": full_metadata.get("patent_number"),
                            "date": full_metadata.get("date"),
                            "ucid": full_metadata.get("ucid"),
                            "classification_main": str(
                                full_metadata.get("classifications", {}).get("main")
                            ),
                            "classification_further": str(
                                full_metadata.get("classifications", {}).get("further")
                            ),
                        }
                    )

            return Document(text=content, metadata=metadata)

        except Exception as e:
            self.logger.error(f"Error processing {md_file}: {str(e)}")
            return None

    def stream_documents(self) -> Iterator[tuple[str, Document]]:
        """Stream documents one at a time, with checkpointing."""
        content_dir = self.data_dir / "content"
        metadata_dir = self.data_dir / "metadata"

        processed_ids, embedded_ids = self.load_checkpoints()

        # Get files that haven't been processed or embedded
        md_files = [
            f
            for f in content_dir.glob("*.md")
            if f.stem not in processed_ids and f.stem not in embedded_ids
        ]

        for md_file in tqdm(md_files, desc="Processing documents"):
            doc = self.process_patent(md_file, metadata_dir)
            if doc:
                processed_ids.add(md_file.stem)
                if len(processed_ids) % self.batch_size == 0:
                    self.save_checkpoint("processing", processed_ids)
                yield md_file.stem, doc

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=40), stop=stop_after_attempt(3)
    )
    def embed_batch(self, batch: List[Document], vector_store: PineconeVectorStore):
        """Embed and index a batch of documents with retry logic."""
        try:
            self.logger.info(f"Starting embedding for batch of {len(batch)} documents")
            index = VectorStoreIndex.from_documents(
                batch,
                storage_context=StorageContext.from_defaults(vector_store=vector_store),
                show_progress=True,
            )
            return True
        except Exception as e:
            self.logger.error(f"Error embedding batch: {str(e)}")
            raise

    def build_index(self, index_name: str):
        """Build both vector and BM25 indices."""
        try:
            # Connect to Pinecone index
            self.logger.info(f"Connecting to Pinecone index: {index_name}")
            pinecone_index = self.pc.Index(index_name)

            # Get initial stats
            initial_stats = pinecone_index.describe_index_stats()
            self.logger.info(f"Initial Pinecone index stats: {initial_stats}")

            # Create vector store
            vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

            # Collect documents for processing
            documents = []
            current_batch = []
            current_batch_ids = []
            embedded_ids = set()

            for patent_id, doc in self.stream_documents():
                documents.append(doc)  # Save for BM25 index
                current_batch.append(doc)
                current_batch_ids.append(patent_id)

                if len(current_batch) >= self.batch_size:
                    # Index batch in vector store with retry logic
                    self.embed_batch(current_batch, vector_store)
                    embedded_ids.update(current_batch_ids)
                    self.save_checkpoint("embedding", embedded_ids)

                    current_batch = []
                    current_batch_ids = []
                    time.sleep(1)  # Rate limiting

            # Process remaining documents
            if current_batch:
                self.embed_batch(current_batch, vector_store)
                embedded_ids.update(current_batch_ids)
                self.save_checkpoint("embedding", embedded_ids)

            # Initialize indices and retrievers
            self.vector_index = VectorStoreIndex.from_vector_store(vector_store)
            self.bm25_retriever = BM25Retriever.from_defaults(
                nodes=documents, similarity_top_k=5
            )
            self.hybrid_retriever = HybridRetriever(
                vector_retriever=self.vector_index.as_retriever(similarity_top_k=5),
                bm25_retriever=self.bm25_retriever,
            )

            # Initialize reranker
            self.reranker = SentenceTransformerRerank(
                model="BAAI/bge-reranker-base", top_n=4
            )

            final_stats = pinecone_index.describe_index_stats()
            self.logger.info(f"Final Pinecone index stats: {final_stats}")
            return self.vector_index

        except Exception as e:
            self.logger.error(f"Error building index: {str(e)}")
            raise

    def search(self, query: str, config: RetrievalConfig) -> List[NodeWithScore]:
        """Perform search based on specified method and configuration."""
        try:
            if not self.vector_index or not self.bm25_retriever:
                raise ValueError("Indices not initialized. Call build_index first.")

            if config.method == "vector":
                retriever = self.vector_index.as_retriever(
                    similarity_top_k=config.top_k
                )
                nodes = retriever.retrieve(query)
            elif config.method == "bm25":
                nodes = self.bm25_retriever.retrieve(query)
            elif config.method == "hybrid":
                nodes = self.hybrid_retriever.retrieve(query)
            else:
                raise ValueError(f"Unknown retrieval method: {config.method}")

            # Apply reranking if configured
            if config.use_reranking and self.reranker:
                nodes = self.reranker.postprocess_nodes(
                    nodes, query_bundle=QueryBundle(query)
                )

            return nodes[: config.top_k]

        except Exception as e:
            self.logger.error(f"Error performing search: {str(e)}")
            raise

    @staticmethod
    def load_index(
        index_name: str,
        pinecone_api_key: str,
        embed_model: Optional[str] = None,
        data_dir: Optional[str] = None,
    ) -> tuple[VectorStoreIndex, Optional[BM25Retriever], Optional[HybridRetriever]]:
        """
        Load an existing index with BM25 and hybrid search capabilities.

        Args:
            index_name: Name of the Pinecone index
            pinecone_api_key: Pinecone API key
            embed_model: Name of the embedding model
            data_dir: Directory containing patent data for BM25 index

        Returns:
            Tuple of (VectorStoreIndex, BM25Retriever, HybridRetriever)
        """
        try:
            # Initialize Pinecone and vector store
            pc = Pinecone(api_key=pinecone_api_key)
            index = pc.Index(index_name)
            vector_store = PineconeVectorStore(pinecone_index=index)

            if embed_model:
                Settings.embed_model = OpenAIEmbedding(model_name=embed_model)

            # Create vector index
            vector_index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store, show_progress=True
            )

            # Initialize BM25 if data directory is provided
            bm25_retriever = None
            hybrid_retriever = None

            if data_dir:
                try:
                    # Load documents for BM25
                    documents = []
                    content_dir = Path(data_dir) / "content"
                    metadata_dir = Path(data_dir) / "metadata"

                    if not content_dir.exists() or not metadata_dir.exists():
                        logging.warning(
                            f"Data directories not found: {content_dir} or {metadata_dir}"
                        )
                        return vector_index, None, None

                    # Process documents for BM25
                    for md_file in content_dir.glob("*.md"):
                        try:
                            with open(md_file, "r", encoding="utf-8") as f:
                                content = f.read().strip()

                            if not content:
                                continue

                            # Read metadata
                            patent_id = md_file.stem
                            metadata = {"patent_id": patent_id}
                            metadata_file = metadata_dir / f"{patent_id}.json"

                            if metadata_file.exists():
                                with open(metadata_file, "r", encoding="utf-8") as f:
                                    full_metadata = json.load(f)
                                    metadata.update(
                                        {
                                            "patent_number": full_metadata.get(
                                                "patent_number"
                                            ),
                                            "date": full_metadata.get("date"),
                                            "ucid": full_metadata.get("ucid"),
                                            "classification_main": str(
                                                full_metadata.get(
                                                    "classifications", {}
                                                ).get("main")
                                            ),
                                            "classification_further": str(
                                                full_metadata.get(
                                                    "classifications", {}
                                                ).get("further")
                                            ),
                                        }
                                    )

                            documents.append(Document(text=content, metadata=metadata))

                        except Exception as e:
                            logging.error(f"Error processing file {md_file}: {str(e)}")
                            continue

                    if documents:
                        # Initialize BM25 retriever
                        bm25_retriever = BM25Retriever.from_defaults(
                            nodes=documents, similarity_top_k=5
                        )

                        # Initialize hybrid retriever
                        hybrid_retriever = HybridRetriever(
                            vector_retriever=vector_index.as_retriever(
                                similarity_top_k=5
                            ),
                            bm25_retriever=bm25_retriever,
                            top_k=5,
                        )

                        logging.info(
                            f"Successfully initialized BM25 with {len(documents)} documents"
                        )
                    else:
                        logging.warning("No documents processed for BM25")

                except Exception as e:
                    logging.error(f"Error initializing BM25: {str(e)}")
                    return vector_index, None, None

            return vector_index, bm25_retriever, hybrid_retriever

        except Exception as e:
            logging.error(f"Error loading index: {str(e)}")
            raise


def main(
    query: str,
    retrieval_method: str = "hybrid",
    use_reranking: bool = False,
    top_k: int = 5,
    data_dir: str = "patent_data",
    checkpoint_dir: str = "patent_checkpoints",
    pinecone_region: str = "us-west-2",
):
    """
    Main function to run the patent search system.

    Args:
        query: Search query string
        retrieval_method: Search method (vector, bm25, or hybrid)
        use_reranking: Whether to use reranking
        top_k: Number of results to return
        data_dir: Directory containing patent data
        checkpoint_dir: Directory for checkpoints
        pinecone_region: Pinecone region

    Returns:
        List of search results with metadata
    """
    try:
        # Initialize builder
        builder = PatentIndexBuilder(
            data_dir=data_dir,
            embed_model="text-embedding-3-small",
            batch_size=100,
            checkpoint_dir=checkpoint_dir,
            pinecone_region=pinecone_region,
        )

        # Build or load index
        index_name = "patent-search"
        try:
            builder.build_index(index_name)
            # Try to load existing index first
            vector_index, bm25_retriever, hybrid_retriever = (
                PatentIndexBuilder.load_index(
                    index_name=index_name,
                    pinecone_api_key=os.getenv("PINECONE_API_KEY"),
                    embed_model="text-embedding-3-small",
                    data_dir=data_dir,
                )
            )
            builder.vector_index = vector_index
            builder.bm25_retriever = bm25_retriever
            builder.hybrid_retriever = hybrid_retriever

            # Initialize reranker
            builder.reranker = SentenceTransformerRerank(
                model="BAAI/bge-reranker-base", top_n=min(top_k, 4)
            )
        except Exception as e:
            logging.warning(f"Could not load existing index: {e}")
            logging.info("Building new index...")
            builder.build_index(index_name)

        # Configure retrieval
        config = RetrievalConfig(
            method=retrieval_method,
            top_k=top_k,
            use_reranking=use_reranking,
            reranking_model="BAAI/bge-reranker-base",
            rerank_top_n=min(top_k, 4),  # Don't rerank more than 4 documents
        )

        # Perform search
        results = builder.search(query, config)

        # Process and format results
        search_results = []
        for node in results:
            result = {
                "patent_id": node.node.metadata.get("patent_id"),
                "score": node.score,
                "patent_number": node.node.metadata.get("patent_number"),
                "date": node.node.metadata.get("date"),
                "classification_main": node.node.metadata.get("classification_main"),
                "classification_further": node.node.metadata.get(
                    "classification_further"
                ),
                "text_snippet": (
                    node.node.text[:300] + "..."
                    if len(node.node.text) > 300
                    else node.node.text
                ),
            }
            search_results.append(result)

        return search_results

    except Exception as e:
        logging.error(f"Error in main function: {str(e)}", exc_info=True)
        raise


def format_results(results: List[Dict]) -> str:
    """
    Format search results for display.

    Args:
        results: List of search result dictionaries

    Returns:
        Formatted string for display
    """
    output = []
    for i, result in enumerate(results, 1):
        output.append(f"\nResult {i}:")
        output.append(f"Patent ID: {result['patent_id']}")
        output.append(f"Patent Number: {result['patent_number']}")
        output.append(f"Date: {result['date']}")
        output.append(f"Relevance Score: {result['score']:.4f}")
        output.append(f"Classification:")
        output.append(f"  Main: {result['classification_main']}")
        output.append(f"  Further: {result['classification_further']}")
        output.append(f"\nText Preview:")
        output.append(f"{result['text_snippet']}")
        output.append("-" * 100)
    return "\n".join(output)


if __name__ == "__main__":
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Patent Search System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("query", type=str, help="Search query string")
    parser.add_argument(
        "--method",
        type=str,
        choices=["vector", "bm25", "hybrid"],
        default="hybrid",
        help="Search method (default: hybrid)",
    )
    parser.add_argument("--rerank", action="store_true", help="Enable reranking")
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of results to return (default: 5)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="patent_data",
        help="Directory containing patent data (default: patent_data)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="patent_checkpoints",
        help="Directory for checkpoints (default: patent_checkpoints)",
    )
    parser.add_argument(
        "--pinecone-region",
        type=str,
        default="us-east-1",
        help="Pinecone region (default: us-west-2)",
    )

    # Parse arguments
    args = parser.parse_args()

    try:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("patent_search.log"),
                logging.StreamHandler(),
            ],
        )

        # Display configuration
        print("\nSearch Configuration:")
        print("-" * 50)
        print(f"Query: {args.query}")
        print(f"Method: {args.method}")
        print(f"Reranking: {'enabled' if args.rerank else 'disabled'}")
        print(f"Top-K: {args.top_k}")
        print(f"Data Directory: {args.data_dir}")
        print(f"Checkpoint Directory: {args.checkpoint_dir}")
        print(f"Pinecone Region: {args.pinecone_region}")
        print("-" * 50)
        print("\nExecuting search...")

        # Run search
        results = main(
            query=args.query,
            retrieval_method=args.method,
            use_reranking=args.rerank,
            top_k=args.top_k,
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            pinecone_region=args.pinecone_region,
        )

        # Display results
        if results:
            print("\nSearch Results:")
            print("=" * 100)
            print(format_results(results))
            print(f"\nTotal results found: {len(results)}")
        else:
            print("\nNo results found.")

    except Exception as e:
        print(f"\nError: {str(e)}")
        logging.error(f"Error in script execution: {str(e)}", exc_info=True)
        sys.exit(1)

