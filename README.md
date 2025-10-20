# Stanford Cars Classification Using ResNet50 with Progressive Unfreezing

This project tackles the **Stanford Cars Dataset** using **ResNet50** for fine-grained car model classification.  
The task involves recognizing **196 distinct car models** through transfer learning and **progressive unfreezing**.

To improve transfer learning performance, the model is fine-tuned in **three stages** instead of unfreezing the entire network at once:

&nbsp;&nbsp;&nbsp;&nbsp;☢️ **Stage 1:** Train only the classifier head and deep layers  
&nbsp;&nbsp;&nbsp;&nbsp;☢️ **Stage 2:** Unfreeze mid-level layers  
&nbsp;&nbsp;&nbsp;&nbsp;☢️ **Stage 3:** Unfreeze the early backbone layers  

Higher-level features adapt first while preserving stable low-level ones, reducing the risk of *catastrophic forgetting*.

## **Task Objectives**

&nbsp;&nbsp;&nbsp;&nbsp;⚡ Create an ideal training/validation split from the original training set    
&nbsp;&nbsp;&nbsp;&nbsp;⚡ Keep the official test set untouched until the final model evaluation     
&nbsp;&nbsp;&nbsp;&nbsp;⚡ Visualize the dataset for insights on car model distribution and image content  
&nbsp;&nbsp;&nbsp;&nbsp;⚡ Explore target labels so we can identify class imbalances or other patterns     
&nbsp;&nbsp;&nbsp;&nbsp;⚡ Perform a multi-stage training procedure with progressive unfreezing of layers  
&nbsp;&nbsp;&nbsp;&nbsp;⚡ Apply data augmentation to improve generalization and reduce overfitting  
&nbsp;&nbsp;&nbsp;&nbsp;⚡ Evaluate model performance using top-1 accuracy, precision, recall, and f1

## **Task Workflow**

&nbsp;&nbsp;&nbsp;&nbsp;✅ *Exploratory Data Analysis (EDA)*  
&nbsp;&nbsp;&nbsp;&nbsp;✅ *Data Preprocessing*  
&nbsp;&nbsp;&nbsp;&nbsp;✅ *Model Tuning*  

## **Dataset Information**

&nbsp;&nbsp;&nbsp;&nbsp;👨‍🚀 *Class Count:* 196    
&nbsp;&nbsp;&nbsp;&nbsp;👨‍🚀 *Official Train Size:* 8144    
&nbsp;&nbsp;&nbsp;&nbsp;👨‍🚀 *Official Test Size:* 8041    
&nbsp;&nbsp;&nbsp;&nbsp;👨‍🚀 *Image Resolution:* Varies, many high-res    
&nbsp;&nbsp;&nbsp;&nbsp;👨‍🚀 *Task:* Fine-grained classification of car models      
&nbsp;&nbsp;&nbsp;&nbsp;👨‍🚀 *Source:* [Dataset Page on Kaggle](https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder)  

***Note:*** The dataset is not included in this repository. To use it, please download it directly from the Kaggle link above.

## **Model Information**

&nbsp;&nbsp;&nbsp;&nbsp;🦖 *Model:* ResNet50  
&nbsp;&nbsp;&nbsp;&nbsp;🦖 *Parameters:* ~25.6M  
&nbsp;&nbsp;&nbsp;&nbsp;🦖 *Top-1 Accuracy (ImageNet):* ~76%  
&nbsp;&nbsp;&nbsp;&nbsp;🦖 *Recommended Input Resolution:* 224x224  
&nbsp;&nbsp;&nbsp;&nbsp;🦖 *Architecture:* CNN with Residual blocks  
&nbsp;&nbsp;&nbsp;&nbsp;🦖 *Source:* [Paper](https://arxiv.org/pdf/1512.03385)  

## **Results**

| Metric | Value |
|---------|--------|
| **Loss** | 1.3082 |
| **Accuracy** | 0.9117 |
| **Precision** | 0.9134 |
| **Recall** | 0.9117 |
| **F1-Score** | 0.9110 |
| **Inference Time** | 00:01:37 |

&nbsp;&nbsp;&nbsp;&nbsp;🦦 The current setup achieves top-1 accuracy of approximately 91%  
&nbsp;&nbsp;&nbsp;&nbsp;🦦 We have taken 10% of the training set for validation (e.g., trigger early stopping)  
&nbsp;&nbsp;&nbsp;&nbsp;🦦 We can improve accuracy (and other metrics) slighly above current if we use the training set in its entirety.  
&nbsp;&nbsp;&nbsp;&nbsp;🦦 However, we refrain from using the test set until the final eval.  
&nbsp;&nbsp;&nbsp;&nbsp;🦦 This is to ensure an unbiased assessment of the model's generalization performance.  

## **How to Run**
### Option 1 - Run Locally
1. Clone the repository
      ```bash
      git clone https://github.com/CobraJet-429/resnet50-stanford-cars.git 
      cd resnet50-stanford-cars
      ```
2. Download the dataset manually from this [Kaggle link](https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder).
   Place the extracted dataset inside the data/ folder:
   ```bash
   data/
   ├── cars_train/
   ├── cars_test/
   └── devkit/
   ```

4. (Optional) Create and activate a virtual environment

5. Install required packages
      ```bash
      pip install -r requirements.txt
      ```
6. Open the notebook
      ```jupyter notebook
      notebooks/resnet50-stanford-cars.ipynb
      ```
### Option 2 - Run Directly on Kaggle

1. Go to the [Kaggle dataset](https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder) and create a new notebook.

2. Click “File” -> “Import Notebook” and upload `resnet50-stanford-cars.ipynb` from this repo.

3. In the right sidebar, enable GPU accelerator T4 x2.

4. Run the cells.
