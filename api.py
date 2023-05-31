from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset, DatasetDict, load_metric

from fastapi import FastAPI, File, UploadFile
import pandas as pd

app = FastAPI()

ID2LABEL = {6: 'Data Science',
            12: 'HR',
            0: 'Advocate',
            1: 'Arts',
            24: 'Web Designing',
            16: 'Mechanical Engineer',
            22: 'Sales',
            14: 'Health and fitness',
            5: 'Civil Engineer',
            15: 'Java Developer',
            4: 'Business Analyst',
            21: 'SAP Developer',
            2: 'Automation Testing',
            11: 'Electrical Engineering',
            18: 'Operations Manager',
            20: 'Python Developer',
            8: 'DevOps Engineer',
            17: 'Network Security Engineer',
            19: 'PMO',
            7: 'Database',
            13: 'Hadoop',
            10: 'ETL Developer',
            9: 'DotNet Developer',
            3: 'Blockchain',
            23: 'Testing'}

## Dorost/resume
# pipe = pipeline("text-classification", model="./resume", device=-1)
pipe = pipeline("text-classification", model="Dorost/resume", device=-1)

@app.post("/resume/")
def get_resume(file: UploadFile = File(...)):
    test_df = pd.read_csv(file.file)
    test_dict = test_df.to_dict('list')
    test_dataset = Dataset.from_dict(test_dict)

    outs = []
    for out in pipe(KeyDataset(test_dataset, "text"), batch_size=8, truncation="only_first"):
        outs.append(out)

    df_pred = pd.DataFrame(outs)
    df_pred['pred_id'] = df_pred['label'].apply(lambda x: int(x.split("_")[1]))
    df_pred['pred_name'] = df_pred['pred_id'].apply(lambda x:ID2LABEL[x])
    df_pred = df_pred.drop('label', axis=1)

    return df_pred.to_dict('list')
