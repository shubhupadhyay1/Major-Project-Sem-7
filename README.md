# Transformer-Based Stock Prediction Model for Nifty 50

Welcome to the Transformer-Based Stock Prediction repository! This project explores using a transformer model to predict short-term directional trends of stocks in the Nifty 50 index. It specifically focuses on anticipating if a stock will experience a 1% price movement within the first three hours of each trading day. By focusing on minute-level data and three custom-built technical indicators (Bollinger Bands, RSI, and ROC), this model aims to offer an efficient, high-frequency trading tool that can make reliable, data-driven predictions.

---

## 📊 Project Overview

Stock price prediction is notoriously challenging due to the volatility and complexity of financial data. This project takes on that challenge with a fresh approach, employing a transformer model to leverage the strengths of self-attention mechanisms in capturing both short-term and long-term dependencies in stock price movements.

### Key Features:
- **Transformer Architecture**: Instead of traditional LSTMs or CNNs, this project uses a transformer model, optimized for time-series data.
- **High-Frequency Data**: Using minute-level data for precise short-term predictions.
- **Custom Indicators**: Includes three technical indicators (Bollinger Bands, RSI, and ROC) as input features for enhanced interpretability and simplicity.
- **Binary Classification Task**: Predicts directional trends based on 1% price movement thresholds.

---

## 🛠️ Getting Started

Follow these steps to set up and run the project.

### Prerequisites

- Python 3.7+
- Required Libraries: `pandas`, `numpy`, `torch`, `scikit-learn`, `matplotlib`
- (Optional) Jupyter Notebook for interactive exploration

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/transformer-stock-prediction.git
   cd transformer-stock-prediction
   ```

2. **Install Dependencies**:
   Make sure to have a virtual environment (optional but recommended):
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Collection**:
   The data for this project is pulled from Yahoo Finance. Run `data_fetcher.py` (or an equivalent script) to gather and preprocess the data:
   ```bash
   python data_fetcher.py
   ```

   Ensure that the dataset is saved in a structured format under the `data/` directory.

---

## 🔍 Model Overview

The transformer model in this repository is configured as an encoder-only model, suitable for high-frequency time-series data. It processes sequences of stock prices and three technical indicators (Bollinger Bands, RSI, ROC) and predicts the directional trend for each stock.

- **Input Data**: Minute-level OHLCV data with custom-calculated indicators.
- **Model Architecture**: Encoder-only transformer with multi-head attention, dense feed-forward layers, and dropout for regularization.
- **Target**: Binary classification of price movement direction (+1% or -1%) by the end of the third trading hour.

---

## 🚀 Running the Model

1. **Data Preprocessing**:
   Ensure the data is correctly preprocessed. Run the `data_preprocess.py` script to format and normalize data:
   ```bash
   python data_preprocess.py
   ```

2. **Training**:
   To train the model, use the `train.py` script. Customize the hyperparameters directly in the script or by passing arguments:
   ```bash
   python train.py --batch_size 32 --epochs 50
   ```

3. **Evaluation**:
   Evaluate the model's performance using the `evaluate.py` script, which provides accuracy, precision, recall, and directional accuracy metrics:
   ```bash
   python evaluate.py
   ```

4. **Inference**:
   Once trained, you can use the `predict.py` script to make predictions on new data:
   ```bash
   python predict.py --input_file path/to/new_data.csv
   ```

---

## 📈 Results

The model outputs predictions on whether a stock’s price will trend up or down within the specified timeframe. Key metrics include:
- **Accuracy**: Measures overall prediction correctness.
- **Directional Accuracy**: Measures accuracy specific to trend direction (up or down).
- **F1 Score**: Provides a balanced view of precision and recall.

---

## 🔧 Customization

Feel free to experiment with different configurations:
- **Window Length**: Change the look-back window for the technical indicators.
- **Indicators**: Try adding or modifying indicators to see if it improves performance.
- **Hyperparameters**: Tune batch size, learning rate, and number of attention heads in `train.py` to optimize model performance.

---
## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 🙌 Acknowledgments

This project was built as a Capstone Project for the subject 'Major Project(21CSR-435)'. Special thanks to Dr. Preet Kamal for her unending support and guidance.

