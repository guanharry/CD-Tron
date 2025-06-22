**CD-Tron: Leveraging Large Clinical Language Models for Early Detection of Cognitive Decline from Electronic Health Records**

This repository contains the evaluation code for **CD-Tron**, a system that uses large clinical language models to detect cognitive decline from clinical notes. 

---

## Installation

Create your Python virtual environment (optional but recommended), then install dependencies:
```
pip install -r requirements.txt
```
---

## Dataset

The dataset `synthetic_scd_data.xlsx` is provided for reproducibility.  
It is fully synthetic. You can feel free to use your own data.

|Column|Description|
|---|---|
|`sectiontxt`|Clinical note text|
|`label_encoded`|0 = No cognitive decline, 1 = Cognitive decline|

---

## Running the Evaluation

Run the script for cognitive decline detection:

```
CDTron_Run.py
```

The script will:

- Load tokenizer and model (from Hugging Face repo).
    
- Tokenize the dataset
    
- Perform inference
    
- Compute evaluation metrics (accuracy, ROC-AUC, PR-AUC)
    
- Generate ROC, PR, and confusion matrix plots and save as high-resolution figures (pdf).
    
- Save predictions (for positive class, i.e., cognitive decline) into `cdtron_predictions.csv`

---

## Output Examples

The outputs include three figures: 1) ROC Curve; 2) Precision-Recall Curve; 3) Confusion Matrix, and one .csv file that include the predicted probabilities for the positive class. All of them are saved in the "results" folder.

---

## Highlights

We aim to make this repository as convenient and reproducible as possible:

- **Hugging Face model integration:** The pretrained model and tokenizer are hosted on Hugging Face Hub. Users can easily load the model directly with a **simple one-line command**, without any manual downloads.
    
- **Synthetic data included:** The model was originally fine-tuned on private real-world clinical data (which cannot be shared). To demonstrate the full pipeline, we provide synthetic data for the task of cognitive decline detection, allowing users to run the full evaluation easily.
    
- **Ready for real-world testing:** Users can easily replace the synthetic data with their own real-world data to test or extend the model in different clinical scenarios.
---

## Citation

If you find the work is useful, please cite our paper. Your support is greatly appreciated!

@article{guan2025cd,
  title={CD-Tron: Leveraging large clinical language model for early detection of cognitive decline from electronic health records},
  author={Guan, Hao and Novoa-Laurentiev, John and Zhou, Li},
  journal={Journal of Biomedical Informatics},
  pages={104830},
  year={2025},
  publisher={Elsevier}
}