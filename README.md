# Sentiment Analysis with BERT and LIME

## 🧠 Overview

This project performs binary sentiment analysis (positive/negative) on IMDb reviews using a fine-tuned BERT-based model. It also provides explainability using the LIME framework.

## 🔧 Tools & Libraries

- Python
- PyTorch
- Hugging Face Transformers
- Datasets (IMDb)
- LIME (Local Interpretable Model-Agnostic Explanations)
- Matplotlib / Seaborn
- Google Colab

## 📂 Project Structure

```
sentiment-bert/
├── src/
│   ├── models/
│   ├── data/
│   ├── training/
│   ├── evaluate.py
│   ├── analyze.py
│   └── lime_explain.py
├── outputs/
├── best_model/
├── README.md
└── requirements.txt
```

## 📈 Model Performance

The fine-tuned model achieves the following scores on the IMDb test set:

| Metric     | Score |
|------------|-------|
| Accuracy   | 0.89  |
| Precision  | 0.88  |
| Recall     | 0.89  |
| F1 Score   | 0.89  |

### ✅ Confusion Matrix

![Confusion Matrix](./confusion_matrix_annotated.png)

### 📊 Metrics by Class

![Metrics Plot](./metrics_barplot.png)

## 🧪 Explainability with LIME

LIME was used to highlight which words influence the model’s decisions:

```text
"I didn't like the movie. It was too slow and boring, but the ending was good."
```

An HTML file `lime_output.html` is saved after running `lime_explain.py`. You can open it in a browser to explore highlighted words.

## ▶️ How to Run

1. Clone the repository or upload it to Google Colab
2. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```
3. Fine-tune the model:
    ```python
    python src/training/train_with_trainer.py
    ```
4. Evaluate the model:
    ```python
    python src/evaluate.py
    ```
5. Analyze misclassifications:
    ```python
    python src/analyze.py
    ```
6. Visualize prediction explanations:
    ```python
    python src/lime_explain.py
    ```

## 📌 Requirements

- Python ≥ 3.8
- Transformers ≥ 4.31
- Datasets
- Scikit-learn
- Matplotlib
- LIME

---

## 📝 License

MIT License
