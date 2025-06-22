# -*- coding: utf-8 -*-
"""
CD-Tron: Leveraging Large Clinical Language Models for Early Detection of
Cognitive Decline from Electronic Health Records
@author: Hao Guan
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, precision_recall_curve, auc, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm

#############################################################################################
# 1. Load Test Data
file_path = 'synthetic_scd_data.xlsx'
df = pd.read_excel(file_path)

#X_test = df['sectiontxt']
#y_test = df['label_encoded']

X_test = df['sectiontxt'][0:]  #fron index 0, can start from any index
y_test = df['label_encoded'][0:]

# Drop missing values and align
X_test = X_test.dropna()
y_test = y_test[X_test.index]

#############################################################################################
# 2. Load Tokenizer and Model from Local Folders
## Loading the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("HAO-AI/cdtron-cognitive-decline")
model = AutoModelForSequenceClassification.from_pretrained("HAO-AI/cdtron-cognitive-decline")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

#############################################################################################
# 3. Tokenize Test Data
X_test_tokenized = tokenizer(X_test.tolist(), padding=True, truncation=True, max_length=512, return_tensors='pt')
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# 4. Create DataLoader
test_dataset = TensorDataset(X_test_tokenized['input_ids'], X_test_tokenized['attention_mask'], y_test_tensor)
test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=1)

#############################################################################################
# 5. Run Inference
predictions, true_labels, all_logits = [], [], []

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Evaluating"):
        input_ids, attention_mask, labels = [x.to(device) for x in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        all_logits.extend(logits.cpu().numpy())

# Convert logits to probabilities
all_logits = torch.tensor(all_logits)
y_pred_proba = torch.softmax(all_logits, dim=1)[:, 1].numpy()

#############################################################################################
# 6. Evaluation Metrics
accuracy = accuracy_score(true_labels, predictions)
classification_rep = classification_report(true_labels, predictions, digits=4)
roc_auc = roc_auc_score(true_labels, y_pred_proba)
precision, recall, _ = precision_recall_curve(true_labels, y_pred_proba)
pr_auc = auc(recall, precision)

print("Performance Evaluation for the Cognitive Decline Detection Model:")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_rep)
print(f"ROC-AUC Score: {roc_auc:.4f}")
print(f"Precision-Recall AUC: {pr_auc:.4f}")

#############################################################################################
# 7. Plot ROC Curve
fpr, tpr, _ = roc_curve(true_labels, y_pred_proba)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - CD-Tron')
plt.legend(loc="lower right")
plt.savefig('./results/roc_curve.pdf', bbox_inches='tight', dpi=300, format='pdf')

#############################################################################################
# 8. Plot PR Curve
plt.figure()
plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - CD-Tron')
plt.legend(loc="lower left")
plt.savefig('./results/precision_recall_curve.pdf', bbox_inches='tight', dpi=300, format='pdf')

#############################################################################################
# 9. Confusion Matrix
cm = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['True 0', 'True 1'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - CD-Tron')
plt.savefig('./results/confusion_matrix.pdf', bbox_inches='tight', dpi=300, format='pdf')

#############################################################################################
# 10. Save Predictions (Note: only saving one class probability (class 1, the postive one))
results_df = pd.DataFrame({
    'predictions': predictions,
    'probabilities': y_pred_proba
})
results_df.to_csv('./results/cdtron_predictions.csv', index=False)