# COMP472

The project aims to use image training datasets found online to train a CNN model to detect a person's facial expression. 

To run the code:

```
git clone https://github.com/KhaledOwaida/COMP472.git
```
then run these commands to import the necessary libraries: 
```
pip install numpy
```
```
pip install torch
```
```
pip install torchvision
```
```
pip install -U scikit-learn
```
```
pip install seaborn
```
```
pip install matplotlib
```

1) Download the dataset folder in the **Dataset** branch.
2) Run the **train.py** file to start training the model contained in **models.py**. The file processes the images such as resizing them and converting them to tensor format. It takes the file paths of a training and testing dataset. It runs several epochs to train, evaluate, and test the model. The model with the best validation accuracy is then saved as *__best-vgg16.pt__*. 

To run the K-Fold cross validation: run the **trainKfold.py**, which trains the model using the **flattened_dataset** for it be to easier. The dataset is made into 10 splits that train a new model instance each fold on 9 splits and 1 split for the testing. The training process is done as before with pre-processing of the images, evaluation, and testing. Various metrics are computed such as the accuracy, precision, recall, and f1-score. We generate a confusion matrix at the end of each fold as well.

### Methodology
* For data cleaning, we removed any fake images and those not containing an actual human face. The analysis uses a dataset explicitly labeled for gender and age, employing accuracy, precision, recall, and F1 score to assess performance discrepancies across demographic groups. Statistical methods were applied to confirm the significance of observed disparities.

### Key Findings
Initial results indicated notable biases, with female and senior groups showing lower performance metrics compared to their counterparts. Such biases could undermine the model's effectiveness and fairness when deployed in practical settings.

### Mitigation Efforts
To address these issues, we implemented several bias mitigation techniques as shown in the **bias analysis.py**:
- **Data Augmentation**: Enhanced the representation of underrepresented groups in the training data.
These efforts led to improved fairness, as evidenced by increased accuracy and precision for previously underperforming groups.

Datasets (sourced from Kaggle):
1) Jonathan Oheix, Face expression recognition dataset, Available: https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset , License: Unkown
2) FORKYKNIGHT, Facial Emotion recognition, Available: https://www.kaggle.com/datasets/chiragsoni/ferdata , License: Public Domain
