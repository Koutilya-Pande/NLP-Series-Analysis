from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import evaluate

metric = evaluate.load('accuracy')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return metric.compute(predictions=predictions, references=labels)

def get_class_weights(df):
    # Ensure that the 'label' column exists
    if 'label' not in df.columns:
        raise ValueError("The input DataFrame must contain a 'label' column.")
    
    # Convert unique labels to a NumPy array
    unique_labels = np.array(sorted(df['label'].unique()))
    
    # Convert the labels to a NumPy array
    y_labels = df['label'].values  # Use .values to get a NumPy array
    
    # Compute class weights
    class_weights = compute_class_weight("balanced", 
                                         classes=unique_labels, 
                                         y=y_labels)
    return class_weights
