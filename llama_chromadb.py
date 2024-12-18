from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    Settings,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
import chromadb
from pathlib import Path
import json
from tqdm import tqdm
import logging
import os
from typing import List, Optional
from dotenv import load_dotenv, find_dotenv


class PatentIndexBuilder:
    def __init__(
        self,
        data_dir: str,
        embed_model: str = "text-embedding-3-small",
        openai_api_key: Optional[str] = None,
    ):
        """
        Initialize the patent index builder.

        Args:
            data_dir: Directory containing patent content and metadata
            embed_model: OpenAI embedding model to use
            openai_api_key: OpenAI API key (optional if set in environment or .env)
        """
        # Load environment variables from .env file
        load_dotenv(find_dotenv())

        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {data_dir} does not exist")

        # Set up OpenAI API key
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key must be provided either through initialization, environment variable, or .env file"
            )

        # Configure global settings
        embed_model = OpenAIEmbedding(model_name=embed_model, api_key=api_key)
        Settings.embed_model = embed_model
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_patent_data(self) -> List[Document]:
        """Load patent data and create Document objects."""
        documents = []
        content_dir = self.data_dir / "content"
        metadata_dir = self.data_dir / "metadata"

        if not content_dir.exists():
            raise FileNotFoundError(f"Content directory {content_dir} does not exist")

        self.logger.info("Loading patent documents...")
        md_files = list(content_dir.glob("*.md"))

        if not md_files:
            raise ValueError(f"No markdown files found in {content_dir}")

        for md_file in tqdm(md_files):
            patent_id = md_file.stem

            try:
                # Read content
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()

                if not content:
                    self.logger.warning(f"Empty content in {md_file}")
                    continue

                # Read metadata
                metadata = {}
                metadata_file = metadata_dir / f"{patent_id}.json"
                if metadata_file.exists():
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        full_metadata = json.load(f)
                        # Flatten metadata and convert nested structures to strings
                        metadata = {
                            "patent_id": patent_id,
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
                else:
                    self.logger.warning(f"No metadata found for {patent_id}")
                    metadata = {"patent_id": patent_id}

                # Create Document object with flattened metadata
                doc = Document(text=content, metadata=metadata)
                documents.append(doc)

            except Exception as e:
                self.logger.error(f"Error processing {md_file}: {str(e)}")
                continue

        if not documents:
            raise ValueError("No valid documents were loaded")

        self.logger.info(f"Loaded {len(documents)} patent documents")
        return documents

    def build_index(self, persist_dir: str):
        """
        Build and save the patent index using ChromaDB.

        Args:
            persist_dir: Directory to save the ChromaDB database
        """
        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Initialize ChromaDB
            chroma_client = chromadb.PersistentClient(path=str(persist_dir))

            # Reset collection if it exists
            try:
                chroma_client.delete_collection("patents")
            except ValueError:
                pass

            collection = chroma_client.create_collection("patents")

            # Create vector store
            vector_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            # Load documents
            documents = self.load_patent_data()

            # Build index
            self.logger.info("Building index...")
            index = VectorStoreIndex.from_documents(
                documents, storage_context=storage_context, show_progress=True
            )

            self.logger.info(f"Index built and saved with {len(documents)} patents")
            return index

        except Exception as e:
            self.logger.error(f"Error building index: {str(e)}")
            raise

    @staticmethod
    def load_index(persist_dir: str, embed_model: Optional[str] = None):
        """
        Load the saved index.

        Args:
            persist_dir: Directory containing the ChromaDB database
            embed_model: Optional embedding model to use for queries

        Returns:
            VectorStoreIndex: Loaded index object

        Raises:
            FileNotFoundError: If the persist_dir doesn't exist
            Exception: For other errors during loading
        """
        if not Path(persist_dir).exists():
            raise FileNotFoundError(f"Index directory {persist_dir} does not exist")

        try:
            # Initialize ChromaDB client
            chroma_client = chromadb.PersistentClient(path=persist_dir)
            collection = chroma_client.get_collection("patents")

            # Create vector store
            vector_store = ChromaVectorStore(chroma_collection=collection)

            # Update settings if embed_model is provided
            if embed_model:
                Settings.embed_model = OpenAIEmbedding(model_name=embed_model)

            # Load index using from_vector_store instead of from_storage_context
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store, show_progress=True
            )

            return index

        except Exception as e:
            logging.error(f"Error loading index: {str(e)}")
            raise


def main():
    try:
        # Initialize builder
        builder = PatentIndexBuilder(
            data_dir="patent_data", embed_model="text-embedding-3-small"
        )

        # Build and save index
        # index = builder.build_index("patent_index_chroma")
        index = builder.load_index("patent_index_chroma")

        # Create query engine
        query_engine = index.as_query_engine(similarity_top_k=5)

        # Example search
        response = query_engine.query("Battery technology for electric vehicles")

        print("\nSearch Results:")
        print("---------------")
        print(response)
        for node in response.source_nodes:
            print(f"\nPatent ID: {node.metadata['patent_id']}")
            print(f"Score: {node.score:.4f}")
            print(f"Chunk: {node.text}")
            print("-" * 50)

    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise


if __name__ == "__main__":
    main()
