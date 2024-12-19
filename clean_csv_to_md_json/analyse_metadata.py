from pathlib import Path
import json
from tqdm import tqdm


def analyze_metadata_sizes(metadata_dir: str):
    """Find metadata sizes with and without ipcr field from classifications."""
    metadata_path = Path(metadata_dir)
    largest_with_ipcr = 0
    largest_without_ipcr = 0
    largest_file = None
    largest_fields = None

    print("Analyzing metadata files...")

    for metadata_file in tqdm(list(metadata_path.glob("*.json"))):
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                full_metadata = json.load(f)

                # Size with IPCR
                full_size = len(str(full_metadata).encode("utf-8"))

                # Create a copy of metadata and remove only ipcr from classifications
                filtered_metadata = full_metadata.copy()
                if "classifications" in filtered_metadata:
                    filtered_metadata["classifications"] = {
                        k: v
                        for k, v in filtered_metadata["classifications"].items()
                        if k != "ipcr"
                    }

                filtered_size = len(str(filtered_metadata).encode("utf-8"))

                if filtered_size > largest_without_ipcr:
                    largest_without_ipcr = filtered_size
                    largest_with_ipcr = full_size
                    largest_file = metadata_file.name

                    # Calculate sizes of top-level fields
                    largest_fields = {}
                    for field, value in filtered_metadata.items():
                        field_size = len(str(value).encode("utf-8"))
                        largest_fields[field] = field_size

        except Exception as e:
            print(f"Error processing {metadata_file}: {str(e)}")
            continue

    print("\nAnalysis Results:")
    print(f"Largest metadata file: {largest_file}")
    print(f"Size with IPCR: {largest_with_ipcr} bytes")
    print(f"Size without IPCR: {largest_without_ipcr} bytes")
    print(f"Size reduction: {largest_with_ipcr - largest_without_ipcr} bytes")

    if largest_fields:
        print("\nField sizes in largest file (without IPCR):")
        for field, size in sorted(
            largest_fields.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"{field}: {size} bytes")


if __name__ == "__main__":
    analyze_metadata_sizes("patent_data/metadata")
