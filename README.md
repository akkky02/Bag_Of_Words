# SMS Spam Detection Project

This project explores different machine learning pipelines for detecting spam SMS messages using a public dataset from Kaggle.

## Data

The data used consists of 5572 SMS messages labeled as either "ham" (not spam) or "spam". It is loaded into a Pandas DataFrame with columns:

- `message`: The raw text content of the SMS message
- `label`: 0 for "ham", 1 for "spam"  

The dataset is imbalanced with ~87% ham and ~13% spam messages.

## Text Preprocessing

The raw text data is preprocessed using a custom `SpacyPreprocessor` class which handles:

- Tokenization, lowercasing
- Stopword removal
- Punctuation removal  
- Email/URL removal
- Lemmatization/stemming
- Batch processing for efficiency

It allows configuring exactly what operations to apply using parameters. This standardized preprocessing helps clean the text before feature extraction.

## Pipelines

Three main pipelines are evaluated:

1. TF-IDF vectors + XGBoost classifier
2. Engineered features + XGBoost classifier 
3. Combined TF-IDF + engineered features + XGBoost  

The processed text data flows through these pipelines to extract informative features, which are then used to train a classifier to detect spam vs ham messages.

## Results

- Pipeline 1 achieves a 0.928 weighted F1 score 
- Pipeline 2 achieves 0.972 F1 using custom spam indicator features
- Pipeline 3 achieves 0.975 F1 score combining both

Pipeline 2 was chosen for its simplicity while maintaining high performance. The final model achieves 98% test accuracy.

## Usage

The trained Pipeline 2 model is serialized so it can be easily loaded and used to classify new SMS messages:

```python
import joblib
pipeline = joblib.load("pipeline2.pkl")  
preds = pipeline.predict(new_messages)
```

## Future Scope

- Deploy model to a production environment
- Experiment with other models like neural networks
- Build a full spam filtering application

Overall, the project shows how proper text preprocessing and feature engineering can improve ML pipelines for NLP tasks.