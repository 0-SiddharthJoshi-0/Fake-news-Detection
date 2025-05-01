# Fake News Detection

A sophisticated machine learning-based system for detecting fake news using multiple classification models.

## Features

- Multiple model support for comprehensive analysis:
  - Logistic Regression (Simple and fast binary classification)
  - Decision Tree (Tree-based model for clear decision paths)
  - Random Forest (Ensemble of decision trees for robust predictions)
  - Gradient Boosting (Advanced ensemble method for accurate predictions)
  - LSTM (Deep learning model for sequential data analysis)

- Real-time text analysis
- Confidence scores for each prediction
- User-friendly interface
- Multi-model consensus for better accuracy

## Screenshots

### Example 1: Detecting Fake News
![Fake News Detection Example](images/fake.jpg)
In this example, all models confidently classify the input text as fake news, with confidence scores ranging from 73.11% to 100%.

### Example 2: Detecting Real News
![Real News Detection Example](images/real.jpg)
Here, the models unanimously classify the Reuters news article about Indonesia's jet purchase as real news, with high confidence scores across all models.

## Installation

```bash
# Clone the repository
git clone https://github.com/0-SiddharthJoshi-0/Fake-news-Detection

# Navigate to project directory
cd Fake-News-Detection

# Install required dependencies
pip install -r requirements.txt
```

## Training the Models

### 1. Data Preprocessing

First, ensure your dataset is in the correct location:
```bash
# Your dataset should be in the Datasets directory
Datasets/
├── Fake.csv
└── True.csv
```

The preprocessing script will:
- Clean and normalize the text
- Remove URLs, HTML tags, and special characters
- Perform lemmatization
- Create train/test splits

Run the preprocessing script:
```bash
python process_data.py
```

### 2. Training the Models

The project uses an enhanced training script that trains multiple models simultaneously. To train all models:

```bash
python train_decision_tree.py
python train_gradient_boosting.py
python train_logistic_regression.py
python train_random_forest.py
python train_lstm.py
```

This will:
- Train all models (Random Forest, Gradient Boosting, Logistic Regression, Decision Tree)
- Save the trained models in the `models/` directory
- Generate performance metrics and training curves
- Save the TF-IDF vectorizer for text preprocessing

The training process includes:
- Cross-validation
- Hyperparameter optimization
- Early stopping
- Model evaluation metrics (accuracy, precision, recall, F1-score)

### 3. Model Outputs

After training, you'll find the following in the `models/` directory:
- Trained model files (`.pkl` format)
- TF-IDF vectorizer
- Training metrics and curves
- Model performance reports

## Using the Web Interface

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Select one or multiple models for analysis
4. Enter the text you want to analyze in the text input field
5. Click "Analyze Text" to get predictions
6. View results from each selected model with confidence scores

## Model Performance

The models are evaluated on multiple metrics:
- Accuracy: Overall prediction accuracy
- Precision: Ability to avoid false positives
- Recall: Ability to find all positive cases
- F1-Score: Harmonic mean of precision and recall

Training curves and performance metrics are saved in the `models/` directory for analysis.
