from llama_index.core import Document, Settings, VectorStoreIndex, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
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
from llama_index.core.retrievers import VectorIndexRetriever
import sys


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
        """Initialize with same parameters as before"""
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
        Settings.chunk_size = None
        Settings.chunk_overlap = 25

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
        """Embed and index a batch of documents with retry logic and verification."""
        try:
            self.logger.info(f"Starting embedding for batch of {len(batch)} documents")

            # Create index from documents
            index = VectorStoreIndex.from_documents(
                batch, vector_store=vector_store, show_progress=True
            )

            return True
        except Exception as e:
            self.logger.error(f"Error embedding batch: {str(e)}")
            raise

    def build_index(self, index_name: str):
        """Build and save the patent index using Pinecone with batching."""
        try:
            # Connect to existing index
            self.logger.info(f"Connecting to Pinecone index: {index_name}")
            pinecone_index = self.pc.Index(index_name)

            # Get initial stats
            initial_stats = pinecone_index.describe_index_stats()
            self.logger.info(f"Initial Pinecone index stats: {initial_stats}")

            # Create vector store and storage context
            vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

            # Load existing progress
            _, embedded_ids = self.load_checkpoints()
            self.logger.info(
                f"Resuming from {len(embedded_ids)} previously embedded documents"
            )

            # Process and embed in batches
            current_batch = []
            current_batch_ids = []
            total_processed = len(embedded_ids)  # Start from previous progress

            self.logger.info("Starting document streaming and processing...")

            for patent_id, doc in self.stream_documents():
                current_batch.append(doc)
                current_batch_ids.append(patent_id)

                if len(current_batch) >= self.batch_size:
                    self.logger.info(
                        f"Processing batch of {len(current_batch)} documents"
                    )

                    # Create index from batch
                    VectorStoreIndex.from_documents(
                        documents=current_batch,
                        storage_context=StorageContext.from_defaults(
                            vector_store=vector_store
                        ),
                    )

                    # Update progress
                    total_processed += len(current_batch)
                    embedded_ids.update(
                        current_batch_ids
                    )  # Add new IDs to existing set
                    self.logger.info(
                        f"Successfully embedded and indexed batch. Total processed: {total_processed}"
                    )

                    # Verify in Pinecone
                    stats = pinecone_index.describe_index_stats()
                    self.logger.info(f"Pinecone index stats after batch: {stats}")

                    # Save checkpoint with all processed IDs
                    self.save_checkpoint("embedding", embedded_ids)

                    # Clear batch
                    current_batch = []
                    current_batch_ids = []

                    # Rate limiting
                    time.sleep(1)

            # Process remaining documents
            if current_batch:
                self.logger.info(
                    f"Processing final batch of {len(current_batch)} documents"
                )
                VectorStoreIndex.from_documents(
                    documents=current_batch,
                    storage_context=StorageContext.from_defaults(
                        vector_store=vector_store
                    ),
                )

                total_processed += len(current_batch)
                embedded_ids.update(current_batch_ids)  # Add final batch IDs
                self.save_checkpoint("embedding", embedded_ids)
                self.logger.info(
                    f"Successfully embedded and indexed final batch. Total processed: {total_processed}"
                )

            # Get final stats
            final_stats = pinecone_index.describe_index_stats()
            self.logger.info(f"Final Pinecone index stats: {final_stats}")

            self.logger.info(
                f"Completed indexing process. Total documents processed: {total_processed}"
            )
            return vector_store

        except Exception as e:
            self.logger.error(f"Error building index: {str(e)}")
            raise

    @staticmethod
    def load_index(
        index_name: str, pinecone_api_key: str, embed_model: Optional[str] = None
    ):
        """Load existing index (same as before)."""
        try:
            pc = Pinecone(api_key=pinecone_api_key)
            index = pc.Index(index_name)
            vector_store = PineconeVectorStore(pinecone_index=index)

            if embed_model:
                Settings.embed_model = OpenAIEmbedding(model_name=embed_model)

            return VectorStoreIndex.from_vector_store(
                vector_store=vector_store, show_progress=True
            )

        except Exception as e:
            logging.error(f"Error loading index: {str(e)}")
            raise

    def check_duplicate_vectors(
        self, index_name: str, similarity_threshold: float = 0.9999
    ) -> Dict[str, Any]:
        """
        Check for duplicate vectors in the Pinecone index by comparing vector similarities.
        Only vectors with extremely high similarity (near identical) are considered duplicates.

        Args:
            index_name: Name of the Pinecone index to check
            similarity_threshold: Cosine similarity threshold for considering vectors as duplicates
                                (default: 0.9999 for nearly identical vectors)

        Returns:
            Dictionary containing duplicate analysis results
        """
        try:
            self.logger.info(f"Checking for duplicate vectors in index: {index_name}")

            # Connect to the index
            index = self.pc.Index(index_name)

            # Get index statistics
            stats = index.describe_index_stats()
            total_vectors = stats.total_vector_count
            self.logger.info(f"Total vectors in index: {total_vectors}")

            # Initialize collections for tracking duplicates
            duplicate_groups: List[Dict[str, Any]] = []
            processed_ids: Set[str] = set()

            # Create a dummy vector for initial query
            dummy_vector = [0.0] * 1536  # Dimension of text-embedding-3-small

            # Process vectors in batches using query
            batch_size = 100
            for i in tqdm(range(0, total_vectors, batch_size), desc="Checking vectors"):
                # Get a batch of vectors using query
                query_response = index.query(
                    vector=dummy_vector,
                    top_k=batch_size,
                    include_values=True,
                    include_metadata=True,
                )

                # Check each vector in the batch
                for match in query_response.matches:
                    if match.id in processed_ids:
                        continue

                    # Query for similar vectors using the actual vector
                    similar_vectors = index.query(
                        vector=match.values,
                        top_k=10,  # Limit to most similar matches
                        include_metadata=True,
                    )

                    # Find duplicates above threshold
                    duplicates = [
                        {
                            "id": v.id,
                            "score": v.score,
                            "patent_id": (
                                v.metadata.get("patent_id")
                                if hasattr(v, "metadata")
                                else None
                            ),
                            "chunk_index": (
                                v.metadata.get("chunk_index")
                                if hasattr(v, "metadata")
                                else None
                            ),
                        }
                        for v in similar_vectors.matches
                        if v.id != match.id and v.score >= similarity_threshold
                    ]

                    if duplicates:
                        group = {
                            "reference_vector": match.id,
                            "reference_patent": (
                                match.metadata.get("patent_id")
                                if hasattr(match, "metadata")
                                else None
                            ),
                            "duplicates": duplicates,
                            "duplicate_count": len(duplicates),
                        }
                        duplicate_groups.append(group)

                        # Mark all IDs in this group as processed
                        processed_ids.add(match.id)
                        processed_ids.update(d["id"] for d in duplicates)

            # Create summary
            total_duplicates = sum(
                group["duplicate_count"] for group in duplicate_groups
            )

            # Create DataFrame for detailed analysis
            if duplicate_groups:
                detailed_df = pd.DataFrame(
                    [
                        {
                            "reference_vector": group["reference_vector"],
                            "reference_patent": group["reference_patent"],
                            "duplicate_vector": dup["id"],
                            "duplicate_patent": dup["patent_id"],
                            "similarity_score": dup["score"],
                        }
                        for group in duplicate_groups
                        for dup in group["duplicates"]
                    ]
                )
            else:
                detailed_df = pd.DataFrame()

            summary = {
                "total_vectors": total_vectors,
                "duplicate_groups": len(duplicate_groups),
                "total_duplicate_vectors": total_duplicates,
                "duplicate_percentage": (
                    (total_duplicates / total_vectors * 100) if total_vectors > 0 else 0
                ),
            }

            self.logger.info(
                f"Found {total_duplicates} duplicate vectors in {len(duplicate_groups)} groups"
            )

            return {
                "summary": summary,
                "duplicate_groups": duplicate_groups,
                "detailed_df": detailed_df,
            }

        except Exception as e:
            self.logger.error(f"Error checking for duplicate vectors: {str(e)}")
            raise

    def remove_duplicate_vectors(self, index_name: str, dry_run: bool = True) -> None:
        """
        Remove duplicate vectors from the index, keeping the first occurrence.

        Args:
            index_name: Name of the Pinecone index
            dry_run: If True, only show what would be deleted without actually deleting
        """
        try:
            self.logger.info(f"Starting duplicate vector removal (dry_run={dry_run})")

            # Check for duplicates
            duplicate_info = self.check_duplicate_vectors(index_name)

            if not duplicate_info["duplicate_groups"]:
                self.logger.info("No duplicate vectors found. Index is clean.")
                return

            # Connect to index
            index = self.pc.Index(index_name)

            # Collect vectors to delete
            vectors_to_delete = []
            for group in duplicate_info["duplicate_groups"]:
                # Keep the reference vector, delete the duplicates
                vectors_to_delete.extend(d["id"] for d in group["duplicates"])

            self.logger.info(f"Found {len(vectors_to_delete)} vectors to delete")

            if dry_run:
                self.logger.info("Dry run - no vectors will be deleted")
                self.logger.info(f"Would delete vectors: {vectors_to_delete[:10]}...")
            else:
                # Delete in batches
                batch_size = 100
                for i in range(0, len(vectors_to_delete), batch_size):
                    batch = vectors_to_delete[i : i + batch_size]
                    index.delete(ids=batch)
                    self.logger.info(f"Deleted batch of {len(batch)} vectors")

                # Verify results
                post_cleanup_info = self.check_duplicate_vectors(index_name)
                self.logger.info(
                    f"Deduplication complete. Removed {len(vectors_to_delete)} duplicate vectors"
                )

        except Exception as e:
            self.logger.error(f"Error removing duplicate vectors: {str(e)}")
            raise

def main(query):
    try:
        builder = PatentIndexBuilder(
            data_dir="patent_data",
            embed_model="text-embedding-3-small",
            batch_size=100,
            checkpoint_dir="patent_checkpoints",
            pinecone_region="us-west-2",
        )

        # Build index
        index_name = "patent-search"
        # builder.build_index(index_name)

        # builder.check_duplicate_vectors(index_name)
        # builder.remove_duplicates(index_name, strategy="keep_latest")

        # Example query
        index = PatentIndexBuilder.load_index(
            index_name=index_name,
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
            embed_model="text-embedding-3-small",
        )

        query_engine = index.as_query_engine(similarity_top_k=10)
        # response = query_engine.query("Lubricants for joints")
        response = query_engine.query(query)
        # response = query_engine.query("Combination locks")

        # print("\nSearch Results:")
        # print("---------------")
        # print(response)
        ls = []
        node_score_debug = []
        for node in response.source_nodes:
            node_score_debug.append(node.score)
            ls.append(node.metadata['patent_id'])
            # print(f"\nPatent ID: {node.metadata['patent_id']}")
            
            # print(f"\nPatent ID: {node.metadata['patent_id']}")
            # print(f"Score: {node.score:.4f}")
            # print(f"Chunk: {node.text}")
        print(node_score_debug)
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise
    # print(ls)
    return ls # Return the list of patent IDs (top 5 most relevant)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python llama_pinecone.py \"<query>\"")
        sys.exit(1)

    # Get the query from the command-line arguments
    query = sys.argv[1]
    
    # Call the main function with the provided query
    main(query)