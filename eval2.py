import pandas as pd
from llama_pinecone import main
import ast

def read_data(path):
    df = pd.read_csv(path)
    return df
# "eval_data.csv"
def retrieve_ids(path): # run queries, retrieve ids and save
    print("Retrieving ids...")
    df = read_data(path)
    # call rag to retrieve document ids and put in retrieved_ids
    df['retrieved_ids'] = df['query'].apply(main)
    # for element in patent_id, if element in document_ids, change to 1

    # tps will be the sum of the list
    df.to_csv("./final_eval_data.csv")
    # print(df.head())
    return df

def generate_relevance(path): # using retrieved ids, generate relevance column
    print("Generating relevance...")
    df = retrieve_ids(path)
    print(type(df["retrieved_ids"][0]))
    df["retrieved_ids"] = df["retrieved_ids"].apply(str).apply(ast.literal_eval)
    # remove duplicates from retrieved_ids first
    df["retrieved_ids"] = df["retrieved_ids"].apply(lambda x: list(set(x)))
    df["relevance"] = df.apply(
    lambda row: [1 if pid in row["patent_id"] else 0 for pid in row["retrieved_ids"]], axis=1)
    df.to_csv("./relevant_final_eval_data.csv")
    return df

def calc_metrics(row):
    """
    Calculate precision and recall.

    Parameters:
    - true_positives: Number of true positive predictions
    - false_positives: Number of false positive predictions
    - false_negatives: Number of false negative predictions

    Returns:
    - precision: The precision score
    - recall: The recall score
    """
    print("Calculating precision and recall...")
    # retrieved_ids = row["retrieved_ids"]
    ground_truth = row["patent_id"].split(",")
    
    # relevance = ast.literal_eval(row["relevance"])
    relevance = row["relevance"]

    # Calculate total relevant (FN + TP)
    total_relevant = len(ground_truth)
    # print(ground_truth,total_relevant)

    # Calculate TP, FP, FN
    tp = sum(relevance)  # True positives: count of 1s in relevance
    fp = len(relevance) - tp  # False positives: total retrieved - TP
    fn = total_relevant - tp  # False negatives: relevant in ground truth but not retrieved

    # Precision and Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    ap_ls = []
    for i in range(len(relevance)):
        ap_ls.append(sum(relevance[:i+1])/(i+1))
    print(ap_ls)
    ap_value = sum(ap_ls)/len(relevance)
    return precision, recall, f1, ap_value

    
        
def run_eval(path):
    # Read the data
    df = generate_relevance(path)
    # Apply the function to each row and collect precision and recall
    df[["precision", "recall", "f1_score", "ap"]] = df.apply(
    calc_metrics, axis=1, result_type="expand"
)
    df.to_csv("./complete_eval_data.csv") # save
# Calculate average precision and recall
    average_precision = df["precision"].mean()
    average_recall = df["recall"].mean()
    average_f1_score = df["f1_score"].mean()
    map = df["ap"].mean()

    

    print(f"Average Precision: {average_precision:.3f}")
    print(f"Average Recall: {average_recall:.3f}")
    print(f"Average F1-Score: {average_f1_score:.3f}")
    print(f"Mean Average Precision: {map:.3f}")
    return average_precision, average_recall, average_f1_score, map

if __name__ == '__main__':
    # generate_relevance("eval_data.csv")
    run_eval("eval_data.csv")