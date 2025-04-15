# 📚 Corpus Analysis on Atticus Legal Contracts Dataset

This project analyzes the **Atticus legal contracts dataset** using Natural Language Toolkit (NLTK). It computes corpus-level insights such as token metrics, filtered tokens, and bigrams using common NLP techniques.

---

## ⚙️ Installation & Dependencies

### 🔹 1. Install Dependencies
```bash
pip install -r requirements.txt
```

If using Google Colab:
```python
!pip install -r requirements.txt
```

### 🔹 2. Required Python Libraries
- os  
- nltk  
- word_tokenize  
- stopwords  
- bigrams  
- Counter  

---

## ▶️ How to Run the Code

### 🔹 1. Prepare the Dataset
Download the Atticus legal contracts dataset and:
- Unzip and extract `full_contract_txt` into the `CUAD_v1/` directory.

### 🔹 2. Run the Main Script
```bash
python main.py
```

This script:
- Loads NLTK resources  
- Concatenates all contract files into a corpus  
- Tokenizes and filters the text  
- Saves intermediate results (`output.txt`, `tokens.txt`, `word_tokens.txt`, `filtered_word_tokens.txt`, `bigrams.txt`)  
- Generates a summary of results

---

## 📝 Summary of the Results
- Extracted total/unique tokens  
- Calculated type-token ratio  
- Generated filtered tokens and bigrams  
- Saved processed data and summary in respective files  

---

# ✨ Sentence Embedding Models for STS2016 Evaluation

This project evaluates various **sentence embedding models** on the STS2016 dataset, computing semantic similarity scores and Pearson correlation with gold-standard values.

---

## ⚙️ Installation & Dependencies

### 🔹 1. Install Dependencies
```bash
pip install -r requirements.txt
```

If using Google Colab:
```python
!pip install -r requirements.txt
```

### 🔹 2. Required Python Libraries
- numpy  
- scipy  
- transformers  
- sentence-transformers  
- tensorflow-hub  
- torch  
- scikit-learn  

---

## ▶️ How to Run the Code

### 🔹 1. Prepare the Dataset
Download STS2016 dataset and place the following files into `sts2016-english-with-gs-v1.0/`:
- `STS2016.input.*.txt`  
- `STS2016.gs.*.txt`

### 🔹 2. Run the Main Script
```bash
python main.py
```
This script:
- Loads SBERT, RoBERTa, USE, ALBERT, LLaMA-2  
- Embeds sentence pairs  
- Computes cosine similarity and normalizes scores (0–5 scale)  
- Outputs results to `*_SYSTEM_OUT.*.txt`

### 🔹 3. Compute Pearson Correlation
```bash
python compute_pearson.py
```
Compares predictions with gold scores and computes Pearson correlation.

---

## 🧠 Models Used & Parameters

| Model       | Embedding Dim | Pooling | Tokenizer           | Fine-tuned? | Framework | Device |
|-------------|----------------|---------|----------------------|-------------|-----------|--------|
| SBERT       | 384            | Mean    | WordPiece            | No          | PyTorch   | GPU    |
| RoBERTa     | 768            | CLS     | SentencePiece        | No          | PyTorch   | GPU    |
| USE         | 512            | Mean    | TensorFlow Tokenizer | No          | TensorFlow| CPU    |
| ALBERT      | 768            | Mean    | WordPiece            | No          | PyTorch   | GPU    |
| LLaMA-2     | 4096           | Mean    | SentencePiece        | No          | PyTorch   | GPU    |

---

## 📁 File Structure

```
📂 project-directory/
├── 📂 sts2016-english-with-gs-v1.0/   # Dataset folder
├── 📝 requirements.txt                # Dependencies
├── 📝 README.md                       # This file
├── 📜 main.py                         # Main execution script
├── 📜 compute_pearson.py              # Pearson correlation script
├── 📜 *_SYSTEM_OUT.*.txt              # Model output files
```

---

## ❗ Troubleshooting

### 1️⃣ Running Out of Memory?
- Use GPU runtime in Google Colab: `Runtime > Change runtime type > GPU`  
- Reduce batch size for large files  

### 2️⃣ Hugging Face Model Access Issues?
- LLaMA-2 requires Hugging Face login:  
```bash
huggingface-cli login
```
- Ensure access to `meta-llama/Llama-2-7b-chat-hf`

---
