import torch
import pandas as pd
import huggingface_hub
import re
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
import gc
from trl import SFTTrainer, SFTConfig
import transformers

## Remove actions from lines

def remove_actions(text):
    result = re.sub(r'\(.*?\)','',text)
    return result
class CharacterChatbot:
    def __init__(self, 
                 model_path,
                 data_path="/content/NLP-Series-Analysis/data/naruto.csv",
                 huggingface_token=None):
        self.model_path = model_path
        self.data_path = data_path
        self.huggingface_token = huggingface_token
        self.base_model_path = "meta-llama/Llama-3.2-3B-Instruct"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.huggingface_token is not None:
            huggingface_hub.login(self.huggingface_token)

        if huggingface_hub.repo_exists(self.model_path):
            self.model = self.load_model(self.model_path)
        else:
             print(f"Model not found at {self.model_path}. Please check the model path and try again. Training our own model...")
             train_dataset = self.load_dataset()
             self.train_model(self.base_model_path, train_dataset)
             self.model = self.load_model(self.model_path)

    def load_model(self, model_path):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        pipeline = transformers.pipeline("text-generation",
                                        model=model_path,
                                        model_kwargs={"torch_dtype":torch.float16, "quantization_config":bnb_config,}
                                        )
        return pipeline
         



    def load_dataset(self):
        nt_df = pd.read_csv(self.data_path)
        nt_df = nt_df.dropna()
        nt_df['line'] = nt_df['line'].apply(remove_actions)
        nt_df['number_of_words']= nt_df['line'].str.strip().str.split(" ")
        nt_df['number_of_words']= nt_df['number_of_words'].apply(lambda x: len(x))
        nt_df['naruto_response_flag'] = 0
        nt_df.loc[(nt_df['name']=='Naruto')&(nt_df['number_of_words']>4), 'naruto_response_flag'] = 1
        indexes = list(nt_df[(nt_df['naruto_response_flag']==1)& (nt_df.index > 0)].index)
        system_prompt = """ You are Naruto Uzumaki, a character from the anime series Naruto. Your responces should reflect his personality and speech patterns \n """
        prompts = []

        for ind in indexes:
            prompt = system_prompt 
            prompt += nt_df.iloc[ind-1]['line']
            prompt += '\n'
            prompt += nt_df.iloc[ind]['line']
            prompts.append(prompt)

        df= pd.DataFrame({'prompt':prompts})
        dataset = Dataset.from_pandas(df)
        return dataset


    def train_model(self,
              base_model_name_or_path,
              dataset,
              output_dir = "./results",
              per_device_train_batch_size = 1,
              gradient_accumulation_steps = 1,
              optim = "paged_adamw_32bit",
              save_steps = 200,
              logging_steps = 10,
              learning_rate = 2e-4,
              max_grad_norm = 0.3,
              max_steps = 300,
              warmup_ratio = 0.3,
              lr_scheduler_type = "constant",
              ):
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, 
                                                     quantization_config= bnb_config,
                                                     trust_remote_code=True)
        model.config.use_cache = False

        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token

        lora_alpha = 16
        lora_dropout = 0.1
        lora_r=64

        peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_r,
            bias="none",
            task_type="CASUAL_LM"
        )

        training_arguments = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size = per_device_train_batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        optim = optim,
        save_steps = save_steps,
        logging_steps = logging_steps,
        learning_rate = learning_rate,
        fp16= True,
        max_grad_norm = max_grad_norm,
        max_steps = max_steps,
        warmup_ratio = warmup_ratio,
        group_by_length = True,
        lr_scheduler_type = lr_scheduler_type,
        report_to = "none"
        )

        max_seq_len = 512

        trainer = SFTTrainer(
            model = model,
            train_dataset=dataset,
            peft_config=peft_config,
            dataset_text_field="prompt",
            max_seq_length=max_seq_len,
            tokenizer=tokenizer,
            args = training_arguments,
        )

        trainer.train()

        # Save model 
        trainer.model.save_pretrained("final_ckpt")
        tokenizer.save_pretrained("final_ckpt")

        # Flush memory
        del trainer, model
        gc.collect()

        base_model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path,
                                                          return_dict=True,
                                                          quantization_config=bnb_config,
                                                          torch_dtype = torch.float16,
                                                          device_map = self.device
                                                          )
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)

        model = PeftModel.from_pretrained(base_model,"final_ckpt")
        model.push_to_hub(self.model_path)
        tokenizer.push_to_hub(self.model_path)

        # Flush Memory
        del model, base_model
        gc.collect()

    def chat(self, message, history):
        messages = []
        # Add the system ptomp 
        messages.append({"role":"system","content":""""Your are Naruto from the anime "Naruto". Your responses should reflect his personality and speech patterns \n"""})

        for message_and_respnse in history:
            messages.append({"role":"user","content":message_and_respnse[0]})
            messages.append({"role":"assistant","content":message_and_respnse[1]})
        
        messages.append({"role":"user","content":message})

        terminator = [
            self.model.tokenizer.eos_token_id,
            self.model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        output = self.model(
            messages,
            max_length=256,
            eos_token_id=terminator,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )

        output_message = output[0]['generated_text'][-1]
        return output_message

