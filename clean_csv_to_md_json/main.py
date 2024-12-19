import pandas as pd
import textwrap
import json
import os
from pathlib import Path
import ast
import re
from typing import Dict, Any, Optional


def extract_important_patent_columns(input_file, output_file):
    """
    Extract important columns from patent data CSV for RAG system.

    Parameters:
    input_file (str): Path to input CSV file
    output_file (str): Path to output CSV file

    Returns:
    pd.DataFrame: Processed dataframe with important columns
    """
    # Important columns for RAG system
    important_columns = [
        # Core content columns
        "abstract",
        "description",
        "claims",
        "invention-title",
        # Classification columns
        "main-classification",
        "further-classification",
        "classification-ipcr",
        # Reference columns
        "citations",
        "priority-claims",
        # Temporal information
        "date",
        # Identifier columns for reference
        "doc-number",
        "ucid",
    ]

    try:
        # Read the CSV file
        df = pd.read_csv(input_file)

        # Check which important columns actually exist in the dataset
        existing_columns = [col for col in important_columns if col in df.columns]
        missing_columns = [col for col in important_columns if col not in df.columns]

        # Extract only the existing important columns
        df_important = df[existing_columns]

        # Save to new CSV file
        df_important.to_csv(output_file, index=False)

        # Print summary
        print("Successfully processed the patent data:")
        print(f"- Input file: {input_file}")
        print(f"- Output file: {output_file}")
        print(f"- Columns extracted: {len(existing_columns)}")
        if missing_columns:
            print("- Warning: The following columns were not found in the input file:")
            for col in missing_columns:
                print(f"  - {col}")

        return df_important

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return None
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None


def view_patent_data(file_path, num_samples=3):
    """
    Display sample patent records in a readable format.

    Parameters:
    file_path (str): Path to the CSV file
    num_samples (int): Number of samples to display
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Get sample rows
        samples = df.sample(n=min(num_samples, len(df)))

        print(f"\nDisplaying {num_samples} sample patents from {file_path}")
        print("=" * 100)

        for idx, row in samples.iterrows():
            print(f"\nPATENT RECORD #{idx}")
            print("-" * 100)

            # Print each field with proper formatting
            for column in df.columns:
                value = row[column]
                if pd.isna(value):
                    value = "[MISSING]"
                elif isinstance(value, str) and len(value) > 100:
                    # Wrap long text fields
                    value = textwrap.fill(
                        value, width=95, initial_indent="    ", subsequent_indent="    "
                    )

                print(f"{column}:")
                print(f"{value}")
                print()

            print("=" * 100)

        # Print summary statistics
        print("\nDATASET SUMMARY:")
        print(f"Total number of records: {len(df)}")
        print(f"Number of columns: {len(df.columns)}")
        print("\nColumn names:")
        for col in df.columns:
            print(f"- {col}")

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error reading file: {str(e)}")


def analyze_patent_missing_data(file_path):
    """
    Analyze missing data percentages in patent dataset and save results.
    Considers both NaN values and empty lists/arrays as missing data.

    Parameters:
    file_path (str): Path to the CSV file containing patent data

    Returns:
    pandas.DataFrame: DataFrame with missing data statistics
    """
    import pandas as pd
    import ast

    # Define important columns to analyze
    important_columns = [
        # Core content columns
        "abstract",
        "description",
        "claims",
        "invention-title",
        # Classification columns
        "main-classification",
        "further-classification",
        "classification-ipcr",
        # Reference columns
        "citations",
        "priority-claims",
        # Temporal information
        "date",
        # Identifier columns for reference
        "doc-number",
        "ucid",
    ]

    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Initialize dictionary to store results
        missing_stats = {}

        # Calculate missing percentage for each column
        for col in important_columns:
            if col in df.columns:
                total_count = len(df)

                # Function to check if value is effectively empty
                def is_empty(val):
                    if pd.isna(val):
                        return True
                    if isinstance(val, str):
                        # Try to parse string as list/array
                        try:
                            parsed_val = ast.literal_eval(val)
                            if (
                                isinstance(parsed_val, (list, dict))
                                and len(parsed_val) == 0
                            ):
                                return True
                        except (ValueError, SyntaxError):
                            # If can't parse as list/array, check if it's empty string
                            return val.strip() == ""
                    return False

                # Count missing values including empty lists
                missing_count = df[col].apply(is_empty).sum()
                missing_percentage = (missing_count / total_count) * 100

                missing_stats[col] = {
                    "total_rows": total_count,
                    "missing_count": missing_count,
                    "missing_percentage": round(missing_percentage, 2),
                }
            else:
                missing_stats[col] = {
                    "total_rows": 0,
                    "missing_count": 0,
                    "missing_percentage": 100.0,
                    "note": "Column not found in dataset",
                }

        # Convert results to DataFrame and sort
        results_df = pd.DataFrame.from_dict(missing_stats, orient="index")
        results_df = results_df.sort_values("missing_percentage", ascending=False)

        # Save results
        results_df.to_csv("missing_data_analysis.csv")

        return results_df

    except FileNotFoundError:
        print(f"Error: Could not find the file {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


def clean_text(text: str) -> str:
    """Clean text by removing special characters and extra whitespace."""
    if pd.isna(text) or text == "[]":
        return ""

    # Handle case where text is a string representation of a list
    if isinstance(text, str) and text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                text = " ".join(str(item) for item in parsed)
        except:
            pass

    # Remove special characters but keep periods and commas
    text = re.sub(r"[^\w\s.,]", " ", str(text))
    # Remove extra whitespace
    text = " ".join(text.split())
    return text


def create_patent_markdown(patent: Dict[str, Any]) -> str:
    """Create markdown content for a patent."""
    sections = []

    # Title
    if patent.get("invention-title"):
        sections.append(f"# {clean_text(patent['invention-title'])}")

    # Abstract
    if patent.get("abstract"):
        abstract_text = clean_text(patent["abstract"])
        if abstract_text:
            sections.append(f"## Abstract\n{abstract_text}")

    # Claims
    if patent.get("claims"):
        claims_text = clean_text(patent["claims"])
        if claims_text:
            sections.append(f"## Claims\n{claims_text}")

    # Description
    if patent.get("description"):
        desc_text = clean_text(patent["description"])
        if desc_text:
            sections.append(f"## Description\n{desc_text}")

    return "\n\n".join(sections)


def extract_metadata(patent: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metadata from patent."""
    return {
        "patent_number": patent.get("doc-number", ""),
        "date": patent.get("date", ""),
        "ucid": patent.get("ucid", ""),
        "classifications": {
            "main": patent.get("main-classification", ""),
            "further": patent.get("further-classification", ""),
            "ipcr": patent.get("classification-ipcr", ""),
        },
    }


def generate_patent_files(input_csv: str, output_dir: str):
    """
    Generate JSON metadata and MD content files for each patent.

    Args:
        input_csv: Path to input CSV file
        output_dir: Directory to store output files
    """
    # Create output directories
    base_dir = Path(output_dir)
    content_dir = base_dir / "content"
    metadata_dir = base_dir / "metadata"

    content_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Read CSV file
    df = pd.read_csv(input_csv)

    # Process each patent
    for idx, row in df.iterrows():
        patent_dict = row.to_dict()

        # Get patent number for filename
        patent_number = patent_dict.get("doc-number", f"patent_{idx}")

        # Create markdown content
        md_content = create_patent_markdown(patent_dict)

        # Extract metadata
        metadata = extract_metadata(patent_dict)

        # Only create files if there's actual content
        if md_content.strip():
            # Save markdown content
            md_path = content_dir / f"{patent_number}.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)

            # Save metadata
            json_path = metadata_dir / f"{patent_number}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

    # Create index file with all metadata
    all_metadata = {}
    for json_file in metadata_dir.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            patent_number = json_file.stem
            all_metadata[patent_number] = json.load(f)

    with open(base_dir / "metadata_index.json", "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2)


# Example usage
if __name__ == "__main__":
    input_file = "patent_rag_csv EP1 0-20.csv"  # Replace with your input file path
    output_file = "cleaned_patent_data.csv"  # Replace with desired output file path

    # df_processed = extract_important_patent_columns(input_file, output_file)

    # view_patent_data(output_file, num_samples=10)

    # results = analyze_patent_missing_data(output_file)
    # if results is not None:
    #     print("\nMissing Data Analysis:")
    #     print("=====================")
    #     print(results)

    generate_patent_files(input_csv=output_file, output_dir="patent_data")
