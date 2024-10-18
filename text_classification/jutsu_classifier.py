import torch
import huggingface_hub
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, DataCollatorWithPadding, pipeline
import pandas as pd
from .cleaner import Cleaner
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from datasets import Dataset
from .training_utils import get_class_weights, compute_metrics
from .trainer import CustomTrainer
import gc


class JutsuClassifier():
    def __init__(self, 
                  model_path,
                    data_path=None, 
                    text_column_name='text', 
                    label_column_name='jutsus', 
                    model_name='distilbert/distilbert-base-uncased', 
                    test_size = 0.2,
                    num_label = 3,
                    huggingface_token=None
                    ):
        self.model_path = model_path
        self.data_path = data_path
        self.text_column_name = text_column_name
        self.label_column_name = label_column_name
        self.model_name = model_name
        self.test_size = test_size
        self.num_label = num_label
        self.huggingface_token = huggingface_token
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.huggingface_token = huggingface_token

        if self.huggingface_token is not None:
            huggingface_hub.login(self.huggingface_token)

        self.tokenizer = self.load_tokenizer()

        if not huggingface_hub.repo_exists(self.model_path):
            # If the data path is provided
            if self.data_path is  None:
                raise ValueError("The data path is provided but the model path is not found in Hugging Face Hub. Please provide a valid model path or set the data path to None.")
            
            train_data,test_data = self.load_data(self.data_path)
            train_data_df = train_data.to_pandas()
            test_data_df = test_data.to_pandas()
            all_data = pd.concat([train_data_df, test_data_df]).reset_index(drop=True)
            class_weights = get_class_weights(all_data)

            self.train_model(train_data, test_data, class_weights)

        self.model = self.load_model(self.model_path)
    def train_model(self, train_data, test_data, class_weights):
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name,
                                                                  num_labels=self.num_label,
                                                                  id2label=self.label_dict,
                                                                  )
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        training_args = TrainingArguments(
            output_dir=self.model_path,
            
            learning_rate=2e-4,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=5,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            
            metric_for_best_model="accuracy",
            push_to_hub=True,
            
            )
        
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=test_data,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics= compute_metrics,
            
            
            )
        
        trainer.set_device(self.device)
        trainer.set_class_weights(class_weights)
        trainer.train()

        # Clearing Memory
        del model, trainer
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()

    def preprocess_function(self, tokenizer, examples):
        return tokenizer(examples['text_cleaned'], truncation=True)


    def load_data(self,data_path):
        df = pd.read_json(data_path,lines=True)
        df['jutsu_type_simplified'] = df['jutsu_type'].apply(self.simplify_jutsu)
        df['text'] = df['jutsu_name'] + ". " + df['jutsu_description']
        df[self.label_column_name] = df['jutsu_type_simplified']
        df = df[['text', self.label_column_name]]
        df = df.dropna()

        # Clean the data
        cleaner = Cleaner()
        df['text_cleaned'] = df[self.text_column_name].apply(cleaner.clean)

        # Encode Labels 
        le = preprocessing.LabelEncoder()
        le.fit(df[self.label_column_name].tolist())

        label_dict = {index:label_name for index, label_name in enumerate(le.__dict__['classes_'].tolist())}
        self.label_dict = label_dict
        df['label'] = le.transform(df[self.label_column_name].tolist())

        # Train / Test Split
        test_size = 0.2
        df_train, df_test = train_test_split(df, 
                                            test_size=test_size, 
                                            stratify=df['label'],)
        
        # Conver Pandas to a hugging face dataset
        train_dataset = Dataset.from_pandas(df_train)
        test_dataset = Dataset.from_pandas(df_test)

        # tokenize the dataset
        tokenized_train = train_dataset.map(lambda examples: self.preprocess_function(self.tokenizer, examples),
                                            batched=True)
        tokenized_test = test_dataset.map(lambda examples: self.preprocess_function(self.tokenizer, examples),
                                            batched=True)
        
        return tokenized_train, tokenized_test

    def load_tokenizer(self):
        if huggingface_hub.repo_exists(self.model_path):
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return tokenizer
    
    def simplify_jutsu(self, jutsu):
        if "Genjutsu" in jutsu:
            return "Genjutsu"
        if "Ninjutsu" in jutsu:
            return "Ninjutsu"
        if "Taijutsu" in jutsu:
            return "Taijutsu"
    

    def load_model(self,model_path):
        
        model = pipeline("text-classification",
                         model= model_path, return_all_scores=True)
        return model
    def postprocess_output(self, model_output):
        output = []
        for item in model_output:
            label = max(item, key=lambda x:x['score'])['label']
            output.append(label)
        return output
    def classify_jutsu(self, text):
        model_output = self.model(text)
        prediction = self.postprocess_output(model_output)
        return prediction


        
