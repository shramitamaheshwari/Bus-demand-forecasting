
# 🚌 Bus Seat Demand Forecasting

This project predicts **bus seat demand** for upcoming journeys using historical booking and search data.  
It uses **LightGBM** for regression to forecast the `final_seatcount` for each (source, destination, date of journey) pair.

---

## 📂 Project Structure
```

BUS DEMAND FORECASTING/
│
├── model.py # Main training & prediction script
├── data/ # Folder containing datasets
│ ├── train.csv
│ ├── test.csv
│ └── transactions.csv
└── result/ # Folder containing submissions
  ├── sample_submission.csv
  └── submission.csv


````


---

## 🚀 Features
- Extracts **time-based features** such as:
  - Day of week
  - Month
- Creates **lag-based features** using transaction data 15 days before the journey.
- Trains a **LightGBM Regressor** with 5-fold cross-validation.
- Outputs predictions for the test set in the required submission format.

---

## 📊 Model Details
- **Algorithm:** LightGBM Regressor
- **Evaluation Metric:** RMSE (Root Mean Squared Error)
- **Cross-Validation:** 5-Fold K-Fold
- **Hyperparameters:**
  - `n_estimators = 1000`
  - `learning_rate = 0.05`
  - Early stopping with `50` rounds

---

## 🛠️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/shramitamaheshwari/Bus-demand-forecasting.git
cd bus-demand-forecasting
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add data

Place the following CSV files in the project directory:

* `train.csv`
* `test.csv`
* `transactions.csv`
* `sample_submission.csv`

---

## ▶️ Run the Project

Run the training and prediction script:

```bash
python model.py
```

The script will:

1. Train the model using historical data.
2. Evaluate it via cross-validation.
3. Generate predictions in `submission.csv`.

---

## 📈 Output

* The **cross-validation RMSE** is displayed after training.
* The **final predictions** are saved as:

```
submission.csv
```

in the project root.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to fork the repo and open a pull request.

---

## 📜 License

This project is for educational and hackathon purposes only.
Please ensure you comply with the competition’s rules before sharing data or code.

---

## 👩‍💻 Author

* **Shramita Maheshwari** – [GitHub Profile](https://github.com/shramitamaheshwari)

```
