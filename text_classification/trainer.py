import torch
from torch import nn
from transformers import Trainer

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        # Remove or adjust any parameters that are no longer valid
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get('labels')

        # Forward Propogation
        outputs = model(**inputs)
        logits = outputs.get('logits')
        logits = logits.float()

        # Compute Loss
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(self.class_weights, dtype=torch.float).to(device=self.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    
    def set_class_weights(self, class_weights):
        self.class_weights = class_weights
    def set_device(self, device):
        self.device = device

