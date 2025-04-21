# Singapore Airlines Review Topic Modeling – README

This project focuses on extracting and analyzing topics from airline customer reviews using BERTopic and sentence-level sentiment analysis.

## Structure

- **fine_tune_embedding.py**  
  This script fine-tunes a SentenceTransformer model (`paraphrase-distilroberta-base-v2`) on domain-specific example pairs tailored to airline reviews.  
  **⚠️ Important:** *To ensure full reproducibility, this script should **not** be rerun after the initial training.*  
  The model is saved as a folder (`trained_embedding_model/`) and reused in the main analysis.

- **ADL_Individual Assignment Notebook**  
  This Jupyter Notebook contains the full pipeline for:
  - Loading the pretrained, fine-tuned embedding model
  - Generating document embeddings
  - Performing topic modeling with BERTopic
  - Analyzing trends over time
  - Comparing sentiment scores and average ratings
  - Providing final recommendations

## Reproducibility Notice

For **100% reproducibility**:
- Do **not** execute `fine_tune_embedding.py`, as doing so would generate new embeddings due to the random nature of the training process.
- Always use the saved model directory (`trained_embedding_model/`) when running the main notebook.

