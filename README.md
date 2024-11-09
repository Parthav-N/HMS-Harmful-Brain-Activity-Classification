# HMS Harmful Brain Activity Classification (ResNet50 Model)

This repository contains a solution for the **HMS Harmful Brain Activity Classification** competition hosted on Kaggle. The goal of this competition is to classify EEG spectrograms into one of several categories related to harmful brain activity using deep learning techniques.

## Problem Description

EEG (electroencephalogram) signals are used to monitor brain activity and can help diagnose various neurological conditions, including seizures. In this competition, the task is to classify EEG spectrograms into one of the following classes:

- **Seizure**
- **Lateralized Periodic Discharges (LPD)**
- **Generalized Periodic Discharges (GPD)**
- **Lodger Rhythm Disorder (LRDA)**
- **Generalized Rhythmic Delta Activity (GRDA)**
- **Other**

The dataset consists of EEG signals that have been converted into spectrograms, and the goal is to predict the likelihood that each EEG spectrogram belongs to one of these classes.

## Dataset

The dataset used for training and testing the model is from the **HMS Harmful Brain Activity Classification** competition on Kaggle.

You can access the dataset from [here](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/data).

The dataset contains:
- Training data with **EEG spectrograms** and **labels**.
- Test data with **EEG spectrograms** for prediction.
- A sample submission file to format the predictions for submission.

## Approach

The solution leverages a **ResNet50** convolutional neural network (CNN) architecture, which is a powerful model for image-based tasks. Here's a breakdown of the steps taken to solve the problem:

1. **Data Preprocessing**:
   - The raw EEG data was processed to generate spectrograms, which are essentially time-frequency representations of the EEG signals.
   - The spectrograms were then preprocessed by normalizing the data, handling missing values, and saving them as `.npy` files for efficient loading.

2. **Model Architecture**:
   - A **ResNet50** model was used, which is known for its ability to learn hierarchical features and handle vanishing gradient problems with its residual connections.
   - The model was trained with a combination of regular convolutional blocks and identity blocks (residual connections).
   
3. **Data Augmentation**:
   - Data augmentation techniques were applied to the training data, including **mixup**, **frequency masking** (cutout), and **time masking**, to improve the model's generalization.

4. **Model Training**:
   - The model was trained using the **Kullback-Leibler Divergence** loss function, and the optimizer used was **Adam**.
   - Learning rate scheduling was implemented using a cosine decay function to improve convergence during training.

5. **Cross-Validation**:
   - **Stratified Group K-Fold** cross-validation was used to ensure the model generalized well across different patients, preventing overfitting to specific individuals.

6. **Prediction**:
   - Once the model was trained, predictions were made on the test set.
   - The output of the model was a set of probabilities for each of the six classes.

7. **Submission**:
   - The predictions were saved in the format required for submission to the Kaggle competition. The results were stored as probabilities for each class for each test sample.

## Metrics

- The model was evaluated using the **log loss** metric, which is commonly used for multi-class classification problems involving probabilities.
- Accuracy and other classification metrics (e.g., F1-score, precision, recall) can also be computed if ground truth labels are available for the test set.

## Results

The model's performance on the validation set was evaluated, and the best model was selected based on **validation loss**. Predictions were then made on the test dataset and saved in the required submission format.

## Files in the Repository

- `train.ipynb`: The Jupyter Notebook containing the complete model training pipeline, including data processing, model training, and evaluation.
- `best_model.keras`: The saved weights of the best-performing model based on validation loss.
- `submission.csv`: The final output file containing predictions for the test set, ready for submission.

## Future Improvements

While this approach achieved reasonable results, further improvements could be made by:
- Experimenting with different architectures, such as more advanced pre-trained models (e.g., **EfficientNet**, **DenseNet**).
- Using additional data or features, such as raw EEG signals or other domain-specific techniques for better feature extraction.
- Fine-tuning hyperparameters (e.g., batch size, learning rate, optimizer choices).
- Applying more aggressive augmentation techniques or combining models in an ensemble.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

