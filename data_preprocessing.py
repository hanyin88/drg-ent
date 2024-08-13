import json
import pandas as pd
from sklearn.model_selection import train_test_split

# a funciton to convert id to labels
def id2label(data):
    # read the data which has two colums: hospital_course and drg
    dc_drg = pd.read_csv(data)

    # remove drg with less than 2 observations
    dc_drg = dc_drg.groupby("drg").filter(lambda x: len(x) >= 2)
    
    # number of unique drg_34_code in dc_drg: 738
    drg_count = dc_drg.drg.nunique()
    
    # rank dc_drg by drg_34_code
    dc_drg = dc_drg.sort_values(by=["drg"])
    
    # make a new dataframe called id2label, where the first column is drg and the second column is the rank of drg starting form 0 to 737
    id2label = pd.DataFrame(dc_drg["drg"].drop_duplicates())
    id2label["label"] = range(0, drg_count)
    
    # in dc_drg, create a new column called label, which is the mapped value from id2label where the key is drg_34_code
    dc_drg["label"] = dc_drg["drg"].map(dict(zip(id2label.drg, id2label.label)))
    
    # split dc_drc into train and test, test takes 10% of the data, set radoom state to 42, stratify by label
    train, test = train_test_split(dc_drg, test_size=0.5, random_state=42, stratify=dc_drg.label)
    
    # rename hospital_course to text, remove column of hadm_id and drg_34_code, and save train and test to csv
    train = train.rename(columns={"hospital_course": "text"})
    test = test.rename(columns={"hospital_course": "text"})
    train = train[["text", "label"]]
    test = test[["text", "label"]]

    return train, test, id2label

if __name__ == "__main__":
    # Read path from the json file
    with open('paths.json', 'r') as f:
        path = json.load(f)
        raw_data_path = path["raw_data_path"]
        train_set_path = path["train_set_path"]
        test_set_path = path["test_set_path"]
        id2label_path = path["id2label_path"]
      
    train, test, id2label = id2label(raw_data_path)

    id2label.to_csv(id2label_path, index=False)
    train.to_csv(train_set_path, index=False)
    test.to_csv(test_set_path, index=False)