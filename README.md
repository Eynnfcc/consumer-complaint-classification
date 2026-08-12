# 🤖 Consumer Complaint Classification using NLP & RoBERTa

An end-to-end **Natural Language Processing (NLP)** project that automatically classifies consumer complaints into financial service categories using both traditional deep learning sequence models and a transformer-based **RoBERTa** model.

The project covers the complete NLP workflow:

**Data Processing → Text Preprocessing → Model Training → Transformer Fine-Tuning → Evaluation → Streamlit Deployment**

---

## 🚀 Project Overview

Financial institutions receive large volumes of consumer complaints every day.

Manually categorizing these complaints can be time-consuming and inconsistent.

This project explores how NLP and deep learning can automate that process by analyzing complaint text and predicting the most appropriate financial service category.

The system compares traditional sequence-based neural networks such as:

- SimpleRNN
- LSTM
- GRU

with a modern transformer-based architecture:

- RoBERTa

The final application allows users to enter a complaint through a **Streamlit interface** and receive a predicted complaint category in real time.

---

## 🎯 Project Objective

The main objective is to build a complete text classification pipeline capable of:

- Processing raw complaint text
- Cleaning and preparing textual data
- Converting text into model-ready representations
- Training deep learning sequence models
- Fine-tuning a transformer model
- Evaluating model performance
- Deploying the final classifier through an interactive application

The project also provides a practical comparison between traditional recurrent neural networks and modern transformer architectures.

---

## 🧹 NLP Preprocessing Pipeline

The preprocessing stage prepares raw complaint text before model training.

Typical processing includes:

- Text cleaning
- Handling missing values
- Removing unnecessary characters
- Normalizing text
- Preparing labels
- Tokenization
- Sequence preparation
- Train/test splitting
- Model-ready feature generation

Different preprocessing pipelines are used where appropriate for:

**Traditional Deep Learning Models** and **Transformer Models**

---

## 🧠 Deep Learning Models

Several neural network architectures were explored for complaint classification.

### 1. SimpleRNN

A recurrent neural network used as a baseline sequence model.

It processes text sequentially and learns patterns from the order of words.

---

### 2. LSTM

Long Short-Term Memory networks improve on standard RNNs by maintaining information over longer sequences.

This makes them more suitable for text where important contextual information may appear far apart.

---

### 3. GRU

Gated Recurrent Units provide a more lightweight alternative to LSTM while still handling long-term dependencies.

GRUs can provide strong sequence-modeling performance with fewer parameters.

---

## 🤖 RoBERTa Transformer

The project also uses **RoBERTa**, a transformer-based language model, for more advanced contextual text understanding.

RoBERTa is fine-tuned on the consumer complaint classification task using the Hugging Face ecosystem.

The transformer pipeline includes:

- RoBERTa tokenizer
- Complaint text tokenization
- Attention masks
- Transformer fine-tuning
- Classification head
- Model evaluation
- Prediction on unseen complaint text

Unlike traditional sequence models, transformers can capture contextual relationships between words across the entire input sequence.

---

## 📊 Model Evaluation

Model performance is evaluated using several classification metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

These metrics help evaluate not only overall prediction accuracy but also performance across individual complaint categories.

This is especially important when working with real-world datasets that may contain **class imbalance**.

---

## 🎨 Streamlit Application

A Streamlit web application provides an interactive interface for real-time predictions.

The user can:

1. Enter a consumer complaint
2. Submit the complaint to the trained model
3. Receive a predicted financial complaint category

This turns the project from a model-development exercise into a complete end-to-end NLP application.

---

## ⚙️ Engineering Highlights

- ✅ End-to-end NLP classification workflow
- ✅ Consumer complaint text preprocessing
- ✅ Sequence preparation for deep learning
- ✅ SimpleRNN implementation
- ✅ LSTM implementation
- ✅ GRU implementation
- ✅ RoBERTa transformer fine-tuning
- ✅ Hugging Face integration
- ✅ Multi-metric model evaluation
- ✅ Confusion matrix analysis
- ✅ Real-time prediction interface with Streamlit
- ✅ Separation of preprocessing, modeling, testing, and application logic
- ✅ Reusable model-loading utilities

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| **Python** | Core project development |
| **TensorFlow / Keras** | RNN, LSTM, and GRU model development |
| **Hugging Face Transformers** | RoBERTa fine-tuning |
| **RoBERTa** | Contextual text classification |
| **Scikit-learn** | Evaluation metrics and utilities |
| **Pandas** | Data manipulation |
| **NumPy** | Numerical processing |
| **Streamlit** | Interactive web application |
| **Git / GitHub** | Version control and project hosting |

---

## 📁 Project Structure

```text
consumer-complaint-classification/
│
├── .gitignore
│
├── README.md
│
├── app.py
│   └── Streamlit application for real-time complaint prediction
│
├── main.py
│   └── Main workflow for traditional NLP / deep learning pipeline
│
├── model.py
│   └── Deep learning model definitions
│
├── preprocessing.py
│   └── Text preprocessing pipeline
│
├── saved_model.py
│   └── Model saving/loading utilities
│
├── testmodel.py
│   └── Model testing and evaluation
│
├── transformer_main.py
│   └── Main workflow for transformer training and evaluation
│
├── transformer_model.py
│   └── RoBERTa / transformer model logic
│
├── transformer_preprocessing.py
│   └── Transformer-specific tokenization and preprocessing
│
├── transformer_utils.py
│   └── Transformer helper utilities
│
└── utils.py
    └── General helper functions
```

---

## 🔄 System Workflow

```text
                     ┌──────────────────────┐
                     │ Consumer Complaints  │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │ Text Preprocessing   │
                     └──────────┬───────────┘
                                │
                 ┌──────────────┴──────────────┐
                 │                             │
                 ▼                             ▼
       ┌─────────────────────┐       ┌─────────────────────┐
       │ Traditional Models  │       │ Transformer Pipeline│
       │ RNN / LSTM / GRU    │       │      RoBERTa        │
       └──────────┬──────────┘       └──────────┬──────────┘
                  │                             │
                  └──────────────┬──────────────┘
                                 │
                                 ▼
                      ┌─────────────────────┐
                      │ Model Evaluation    │
                      └──────────┬──────────┘
                                 │
                                 ▼
                      ┌─────────────────────┐
                      │ Best Model / Output │
                      └──────────┬──────────┘
                                 │
                                 ▼
                      ┌─────────────────────┐
                      │ Streamlit Web App   │
                      └─────────────────────┘
```

---

## 💻 Installation

### 1. Clone the repository

```bash
git clone <YOUR-REPOSITORY-URL>
cd consumer-complaint-classification
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

**Windows**

```bash
venv\Scripts\activate
```

**macOS / Linux**

```bash
source venv/bin/activate
```

### 4. Install the required dependencies

Install the libraries required by the project, including:

```bash
pip install pandas numpy scikit-learn tensorflow transformers torch streamlit
```

If you later add a `requirements.txt`, use:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

To launch the Streamlit interface:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

---

## 💡 Example Use Case

A user enters a complaint such as:

```text
I have been charged multiple times for a transaction that I did not make,
and my bank has not resolved the issue.
```

The NLP model processes the complaint and predicts the most appropriate financial service category.

This type of system could help organizations automatically route large volumes of customer complaints to the correct department.

---

## 🏆 What I Learned

This project provided practical experience across the complete NLP development lifecycle.

Key areas included:

- Natural Language Processing
- Text preprocessing
- Tokenization
- Sequence modeling
- SimpleRNN architecture
- LSTM architecture
- GRU architecture
- Transformer architecture
- RoBERTa fine-tuning
- Hugging Face Transformers
- Model evaluation
- Class imbalance challenges
- Streamlit deployment
- End-to-end AI system development

One of the most valuable parts of the project was comparing **traditional recurrent neural networks with modern transformer models** and understanding how contextual language representations improve text classification.

---

## 🔮 Future Improvements

Potential future improvements include:

- Hyperparameter optimization
- More advanced class-balancing strategies
- Model comparison dashboard
- Probability/confidence scores for predictions
- Explainable AI techniques
- Batch complaint classification
- REST API deployment
- Docker containerization
- Cloud deployment
- Additional transformer architectures such as DeBERTa or DistilRoBERTa
- Experiment tracking with MLflow or Weights & Biases

---

## 👨‍💻 Author

Developed as part of my journey in:

**Artificial Intelligence • Machine Learning • NLP • Deep Learning • Transformers**

---

## ⭐ Support

If you find this project interesting, consider giving the repository a ⭐.

Feedback and suggestions are welcome.
- Pandas
- NumPy

## Project Structure
