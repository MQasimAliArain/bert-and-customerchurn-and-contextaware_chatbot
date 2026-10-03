## News Topic Classifier Using BERT
To fine-tune a pre-trained Transformer model (**BERT**) to automatically categorize news headlines into four distinct topic categories: World, Sports, Business, and Sci/Tech.

### Dataset
* **Source:** AG News Dataset (via Hugging Face Datasets).
* **Features:** Raw News Headlines (Text).
* **Target:** Topic Category (4 classes).

### Models Used
* **BERT-base-uncased:** A state-of-the-art transformer model used for sequence classification through transfer learning.
* **Gradio Interface:** A web-based deployment tool used to create a live, interactive environment for real-time headline prediction.

### Methodology
* **Preprocessing:** Data was tokenized using the BERT fast tokenizer with truncation and padding to a 128-token limit.
* **Optimization:** Training was conducted on a 10,000-sample subset for 1 epoch, specifically optimized for **CPU-based environments** using the `accelerate` library.
* **Deployment:** A functional pipeline was integrated into a Gradio UI to allow for immediate inference and user interaction.

### Results
The model achieved high predictive performance, reaching over **90% accuracy** on the test subset. Evaluation via a **Normalized Confusion Matrix** confirmed the model's ability to distinguish between diverse categories, with particularly strong performance in Sports and World News. The live demo successfully demonstrates real-time classification with high confidence scores.

### Skills Gained
* NLP architecture using Transformers.
* Transfer learning and model fine-tuning.
* Evaluation metrics for multi-class text classification (Accuracy & F1-Score).
* Model deployment for live interaction.

## End-to-End ML Pipeline with Scikit-learn Pipeline API

To develop an end-to-end Machine Learning pipeline that predicts the likelihood of customer attrition (churn) and provides an interactive interface for real-time risk assessment.

### Dataset
* **Source:** IBM Telco Customer Churn Dataset.
* **Features:** 19 columns including Demographic (Gender, Seniority), Services (Internet, Tech Support), and Financial (Monthly Charges, Contract Type).
* **Target:** Churn Status (Binary: Yes/No).

### Models & Tools Used
* **Random Forest Classifier:** An ensemble learning model chosen for its high accuracy and ability to handle non-linear relationships.
* **Scikit-Learn Pipeline:** Integrated preprocessing (Scaling & One-Hot Encoding) into a single `.pkl` object for seamless deployment.
* **IPyWidgets:** A Python-based framework used to build a professional, tabbed interactive dashboard within Jupyter/Colab.

###  Methodology
* **Preprocessing:** Automated handling of `TotalCharges` (string-to-numeric) and median imputation for missing values.
* **Feature Engineering:** One-Hot Encoding for categorical variables and Standard Scaling for financial metrics.
* **Optimization:** Used **GridSearchCV** to tune hyperparameters (`max_depth`, `n_estimators`), ensuring the model generalizes well to new data.
* **Deployment:** Created a 3-tab UI (Personal, Services, Billing) with live probability scoring and color-coded risk alerts.

###  Results
The model successfully identifies the primary drivers of churn, specifically **Contract Type** and **Tenure**.
* **Accuracy:** Achieved high cross-validation scores through systematic hyperparameter tuning.
* **Evaluation:** Confirmed via a **Confusion Matrix** and **Classification Report**, showing strong precision in detecting at-risk customers.
* **User Experience:** The final dashboard allows for instant "what-if" analysis by changing customer attributes.

### Skills Gained
* **End-to-End Pipeline Design:** Building robust, reproducible ML workflows using Scikit-Learn.
* **Hyperparameter Tuning:** Systematic optimization of model parameters via Grid Search.
* **Feature Importance Analysis:** Identifying and interpreting which business variables drive customer behavior.
* **Interactive UI Development:** Designing functional, user-friendly dashboards for data-driven decision-making.

## Context-Aware Chatbot Using LangChain or RAG
To build a high-performance **Retrieval-Augmented Generation (RAG)** pipeline that allows users to interact with PDF documents through a natural language interface, specifically optimized for comparative data analysis and ranking tasks.

## Dataset
* **Source:** User-provided PDF documents (placed in a local `/data` directory).
* **Features:** Unstructured text extracted from multi-page PDF files.
* **Target:** Context-aware responses and data-driven insights.

## Models & Technologies Used
* **Llama-3.3-70b-versatile:** A high-reasoning LLM accessed via **Groq** for near-instant inference.
* **all-MiniLM-L6-v2:** A lightweight **HuggingFace** transformer model used to generate semantic vector embeddings.
* **ChromaDB:** A vector database used for local storage and efficient similarity searches.
* **LangChain (LCEL):** The orchestration framework used to chain the retriever, prompt templates, and LLM together.

## Methodology
* **Preprocessing:** Documents are parsed using `PyPDFLoader` and split into 1000-character chunks with a 200-character overlap using `RecursiveCharacterTextSplitter`.
* **Optimization:** Implemented a **Senior Economic Data Analyst** system prompt to ensure the model performs numerical ranking and cross-country comparisons with high precision.
* **Retrieval:** Uses a similarity-based retriever configured to fetch the top 3 most relevant context windows ($k=3$).
* **UI Deployment:** An interactive, stylized chat interface built with `ipywidgets` and custom HTML/JavaScript to handle auto-scrolling and message history within Jupyter.
  
## Results
The system demonstrates exceptional performance in extracting specific data points from complex documents. By using a **temperature of 0**, the chatbot maintains high factual integrity, making it ideal for professional economic or technical analysis. The integration of **Groq** allows for complex 70B model reasoning with sub-second response times, significantly outperforming standard cloud-based APIs.

## Skills Gained
* **RAG Architecture:** Implementing the full lifecycle of document ingestion, embedding, and retrieval.
* **Prompt Engineering:** Designing specialized personas for structured data extraction and ranking.
* **Vector Databases:** Managing local vector stores with ChromaDB for semantic search.
* **Interactive UI Design:** Creating front-end components within Python environments using `ipywidgets`.


