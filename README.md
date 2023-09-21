# EmotionClassification
Pet project in evaluating different models (both deep learning and traditional) for the NLP task of classifying emotions

## Dataset used:
The dataset used for this project is an [emotion classification dataset from HuggingFace](https://huggingface.co/datasets/dair-ai/emotion), containing twitter messages classified into 6 emotions:
 - **anger**
 - **fear**
 - **joy**
 - **love**
 - **sadness**
 - **surprise**

Preprocessing used:
 - Word Counts
 - TFIDF
 - Resampling to correct for imbalanced data (not used for every)

## Models:
 
 | Model Used | Training Accuracy | Validation Accuracy |
 | :--------- | ------------- | ----- |
 | **Naive Bayes** (resampling + TFIDF) | 0.93 | 0.80 |
 | **Logistic Regression** (with spacy embeddings) | 0.335 | 0.35 |
 | **Simple Decision Trees** (word_counts) | *0.998* | *0.844* |
 | **Gradient Boosted Trees** (word_counts) | 0.997 | 0.834
 | **LSTMs** | --- | 0.8335 |


## LSTM Details
### Neural Network Architecture
$$ Embedding (64) \rightarrow LSTM (32) \rightarrow Linear (6) \rightarrow LogSoftMax $$

### Training History
Loss history | Accuracy history 
------- | -------
![Training History](/images/LSTM_training_losses.png) | ![](/images/LSTM_validation_accuracies.png)
