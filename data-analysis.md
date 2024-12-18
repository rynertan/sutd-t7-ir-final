# Data Analysis

## Initial Columns

['invention-title', 'intention-to-grant-date', 'country', 'lang', 'doc-number', 'claims', 'further-classification', 'inventors', 'applicants', 'citations', 'priority-claims', 'ucid', 'child-doc', 'abstract', 'description', 'family-id', 'ep-contracting-states', 'main-classification', 'agents', 'parent-doc', 'date', 'is_representative', 'kind', 'term-of-grant', 'format', 'status', 'classification-ipcr']

## Important/Chosen Columns

Here are the columns formatted as a markdown list with proper grouping:

### Core content columns

- `abstract`
- `description`
- `claims`
- `invention-title`

### Classification columns

- `main-classification`
- `further-classification`
- `classification-ipcr`

### Reference columns

- `citations`
- `priority-claims`

### Temporal information

- `date`

### Identifier columns

- `doc-number`
- `ucid`

## Missing Data Analysis

Total number of rows: 271356
Rows with any missing data: 219349
Percentage of rows with missing data: 80.83%

| Column                 | Total Rows | Missing Count | Missing Percentage |
| ---------------------- | ---------- | ------------- | ------------------ |
| citations              | 271,356    | 271,356       | 100.00             |
| claims                 | 271,356    | 221,118       | 81.49              |
| description            | 271,356    | 221,081       | 81.47              |
| further-classification | 271,356    | 219,349       | 80.83              |
| main-classification    | 271,356    | 185,128       | 68.22              |
| abstract               | 271,356    | 153,515       | 56.57              |
| classification-ipcr    | 271,356    | 134           | 0.05               |
| invention-title        | 271,356    | 71            | 0.03               |
| priority-claims        | 271,356    | 81            | 0.03               |
| date                   | 271,356    | 0             | 0.00               |
| doc-number             | 271,356    | 0             | 0.00               |
| ucid                   | 271,356    | 0             | 0.00               |

## Creating MD and Metadata JSON

Benefits of separate metadata:

- Can be used for filtering before embedding search
- Available for post-processing results
- Doesn't dilute the semantic meaning of the main content
- Can be used for metadata-based sorting and filtering
- Enables faceted search capabilities

### File Structure

```
patent_data/
├── content/
│   ├── patent1.md
│   ├── patent2.md
│   └── ...
├── metadata/
│   ├── patent1.json
│   ├── patent2.json
│   └── ...
└── metadata_index.json
```

### JSON Format

```
{
  "patent_number": "123456",
  "date": "2024-01-01",
  "ucid": "US123456",
  "classifications": {
    "main": "A61K",
    "further": "B01J",
    "ipcr": "C07D"
  }
}
```

### Markdown Format

```
# Patent Title

## Abstract
Abstract text here

## Claims
Claims text here

## Description
Description text here
```
