import spacy
from nltk.tokenize import sent_tokenize
import pandas as pd
import os
import sys
import pathlib 
from ast import literal_eval
from utils.data_loader import load_subtitles_dataset
folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path,'../'))
from utils import load_subtitles_dataset


class NamedEntityRecognizer:
    def __init__(self):
        self.nlp_model = self.load_model()
        pass
    def load_model(self):
        nlp = spacy.load("en_core_web_trf")
        return nlp
    def get_ners_inference(self,script):
        script_senteces = sent_tokenize(script)
        ner_dict = []
        for sentence in script_senteces:
            doc = self.nlp_model(sentence)
            ners = set()
            for entity in doc.ents:
                if entity.label_ =="PERSON":
                    full_name = entity.text
                    first_name = full_name.split(" ")[0]
                    first_name = first_name.strip()
                    ners.add(first_name)
            ner_dict.append(ners)
        return ner_dict
    def get_ners(self, dataset_path, save_path=None):
        print(f"Loading dataset from: {dataset_path}")  # Print the dataset path
        if save_path is not None and os.path.exists(save_path):
            print(f"Loading NER results from: {save_path}")  # Print the NER results path
            df = pd.read_csv(save_path)
            df['ners'] = df['ners'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
            return df

        # Load Dataset
        df = load_subtitles_dataset(dataset_path)
        print(f"Dataset loaded with {len(df)} records.")  # Print the number of records loaded
       

        # Run Inference
        df['ners'] = df['script'].apply(self.get_ners_inference)

        if save_path is not None:
            df.to_csv(save_path, index=False)
            print(f"NER results saved to: {save_path}")  # Print confirmation of saving results
        return df



