import io
import json
import os
import pandas as pd
from datasets import load_dataset, Dataset
from PIL import Image

def create_chart_qa_with_reasoning_dataset(reasoning_file, output_folder, override=False):
    if os.path.exists(output_folder) and override==False:
        print("Dataset already exists. Set override=True to force override.")
        return 
    reasoning_label_dataframe = pd.read_parquet(reasoning_file)
    
    ds = load_dataset('HuggingFaceM4/ChartQA', split='train')
    
    new_column_df = pd.DataFrame({'reasoning': reasoning_label_dataframe['label']})
    # Convert the DataFrame to a dictionary
    new_column_dict = new_column_df.to_dict(orient='records')
    # Add the new column to the dataset
    modified_dataset = ds.map(lambda examples, idx: {'reasoning': new_column_dict[idx]['reasoning']}, with_indices=True)

    def transform(batch):
        batch['label'] = f"{batch['reasoning']} {batch['label'][0]}"
        return batch

    modified_dataset = modified_dataset.map(transform)

    modified_dataset.remove_columns(['reasoning'])
    
    # Step 4: Convert the modified DataFrame back to a Hugging Face dataset
    os.makedirs(f"{output_folder}/train/", exist_ok=True)
    modified_dataset.to_parquet(f"{output_folder}/train/data.parquet")
    
    meta_data = {"splits":['train']}
    with open(f'{output_folder}/dataset_dict.json', 'w') as f:
        f.write(json.dumps(meta_data, indent=4))