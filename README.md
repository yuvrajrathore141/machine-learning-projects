# Machine Learning, Deep Learning, RL & Generative AI Portfolio

A comprehensive collection of 9 end-to-end Machine Learning, Computer Vision, Reinforcement Learning, Deployment, and Generative AI (RAG) projects.

---

## 📁 Repository Overview

Each project is self-contained within its own directory, complete with data preprocessing, model development, evaluation metrics, and visualizations.

```
.
├── Project_01_adult_income_classification/
├── Project_02_cifar10_cnn/
├── Project_03_lfw_face_recognition/
├── Project_04_mri_cancer_detection/
├── Project_05_cartpole_rl/
├── Project_06_lunar_lander_rl/
├── Project_07_movie_recommendation_system/
├── Project_08_render_deployment/
└── Project_09_rag_chatbot_capstone/
```

---

## 🚀 Projects Summary

### 1. [Project 01 - Adult Census Income Classification](./Project_01_adult_income_classification/)
- **Problem Statement**: Predict whether an individual's annual income exceeds $50,000 based on demographic and employment attributes.
- **Dataset**: UCI Adult Census Income dataset (via OpenML).
- **Tech Stack**: Python, Pandas, Scikit-learn, Seaborn, Matplotlib.
- **Key Techniques**: Data cleaning, categorical one-hot encoding, class imbalance handling, Logistic Regression & Random Forest modeling, ROC-AUC evaluation.

### 2. [Project 02 - CIFAR-10 Image Classification using CNN](./Project_02_cifar10_cnn/)
- **Problem Statement**: Build and train a Convolutional Neural Network (CNN) from scratch to classify RGB images across 10 object classes.
- **Dataset**: CIFAR-10 (60,000 32x32 color images across 10 balanced classes).
- **Tech Stack**: TensorFlow / Keras, NumPy, Matplotlib.
- **Key Techniques**: Convolutional layers, Max Pooling, Batch Normalization, Dropout regularization, Data Augmentation.

### 3. [Project 03 - Face Recognition using CNN (LFW Dataset)](./Project_03_lfw_face_recognition/)
- **Problem Statement**: Perform facial recognition under unconstrained ("in the wild") conditions.
- **Dataset**: Labeled Faces in the Wild (LFW) dataset.
- **Tech Stack**: OpenCV, Scikit-learn, TensorFlow / PyTorch, Matplotlib.
- **Key Techniques**: Face detection, feature extraction, PCA / CNN embeddings, multiclass identity classification.

### 4. [Project 04 - Brain Tumor MRI Cancer Detection](./Project_04_mri_cancer_detection/)
- **Problem Statement**: Screen brain MRI scans for the presence of brain tumors with high sensitivity and specificity.
- **Dataset**: Public Brain Tumor MRI Dataset (via KaggleHub).
- **Tech Stack**: PyTorch / Keras, OpenCV, PIL, Scikit-learn.
- **Key Techniques**: Image preprocessing, Transfer Learning (ResNet / EfficientNet), Binary & Multiclass Classification, Sensitivity-Recall optimization.

### 5. [Project 05 - Cart-Pole RL Agent Training](./Project_05_cartpole_rl/)
- **Problem Statement**: Train a reinforcement learning agent to balance a pole upright on a cart using state space observations.
- **Dataset**: Simulated environment experience generated online via Gymnasium `CartPole-v1`.
- **Tech Stack**: PyTorch / TensorFlow, Gymnasium, NumPy.
- **Key Techniques**: Deep Q-Network (DQN), Experience Replay Buffer, $\epsilon$-greedy exploration policy, Target Network updates.

### 6. [Project 06 - Lunar Lander RL Agent Training](./Project_06_lunar_lander_rl/)
- **Problem Statement**: Train an autonomous agent to land a spacecraft safely on a landing pad between flags.
- **Dataset**: Simulated environment experience generated online via Gymnasium `LunarLander-v3`.
- **Tech Stack**: PyTorch, Gymnasium, Box2D, Matplotlib.
- **Key Techniques**: Deep Q-Learning (DQN), Dueling DQN / Double DQN, Reward shaping, Policy evaluation.

### 7. [Project 07 - Movie Recommendation System](./Project_07_movie_recommendation_system/)
- **Problem Statement**: Predict user movie ratings and generate personalized movie recommendations.
- **Dataset**: MovieLens 100K dataset (100,000 ratings from 943 users across 1,682 movies).
- **Tech Stack**: Scikit-learn, Surprise / PyTorch, Pandas, NumPy.
- **Key Techniques**: Collaborative Filtering (Matrix Factorization / SVD), Content-based filtering, Cosine Similarity, RMSE evaluation.

### 8. [Project 08 - End-to-End Render Deployment Microservice](./Project_08_render_deployment/)
- **Problem Statement**: Package and deploy a trained Income Classification ML pipeline as a production web service on Render.
- **Live Service URL**: [https://adult-income-api-6av7.onrender.com](https://adult-income-api-6av7.onrender.com)
- **Dataset**: UCI Adult Income dataset.
- **Tech Stack**: FastAPI, Uvicorn, Scikit-learn, Pydantic, Docker, Render, Pytest/TestClient.
- **Key Features**: REST API with `/health` and `/predict` endpoints, `model.joblib` pipeline serialization, `render.yaml` Infrastructure-as-Code blueprint, `Procfile`, and `Dockerfile`.

### 9. [Project 09 - RAG Chatbot Capstone Project](./Project_09_rag_chatbot_capstone/)
- **Problem Statement**: Build a Retrieval-Augmented Generation (RAG) chatbot that answers domain-specific technical questions using external document retrieval and an LLM generator.
- **Dataset**: 20 Newsgroups technical text corpus.
- **Tech Stack**: LangChain / LlamaIndex / SentenceTransformers, HuggingFace Transformers, FAISS / TF-IDF, Python.
- **Key Techniques**: Document chunking, vector embedding, semantic similarity retrieval, prompt context augmentation, hallucination prevention.

---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yuvrajrathore141/machine-learning-projects.git
   cd machine-learning-projects
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r Project_08_render_deployment/requirements.txt
   ```

---

## 🌐 Live Microservice Deployment (Project 08)

The Project 08 FastAPI Income Classification microservice is deployed live on Render:
- **Base Endpoint & Health**: [https://adult-income-api-6av7.onrender.com/health](https://adult-income-api-6av7.onrender.com/health)
- **Interactive Swagger API Docs**: [https://adult-income-api-6av7.onrender.com/docs](https://adult-income-api-6av7.onrender.com/docs)

### Running Project 08 Service Locally

```bash
cd Project_08_render_deployment
python train.py
python -m uvicorn main:app --reload --port 8000
```
Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser to test the interactive API endpoints.
