**BLOOD GROUP DETECTION USING ML** 

This project uses Machine Learning to detect blood groups from fingerprint images using the **SOCOFing dataset**.  
It includes scripts for data preprocessing, training, and prediction.



## 📂 Project Structure

```

.
├── SOCOFing/               # Dataset folder
├── dataset/                # Processed dataset
├── model/                  # Saved ML model files
├── app.py                  # Main application script
├── dataset\_preparer.py     # Data preprocessing script
├── train\_model.py          # Model training script
├── predict.py              # Prediction script
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation



## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Lohith-Magsani/BLOOD-GROUP-DETECTION-USING-ML.git
cd BLOOD-GROUP-DETECTION-USING-ML
````

### 2️⃣ Install Dependencies

Make sure you have **Python 3.8+** installed, then run:

```bash
pip install -r requirements.txt
```

### 3️⃣ Prepare the Dataset

Place the **SOCOFing** dataset in the project root folder and run:

```bash
python dataset_preparer.py
```

### 4️⃣ Train the Model

```bash
python train_model.py
```

### 5️⃣ Run the Application

```bash
python app.py
```

---

## 🧠 Model Details

* **Algorithm**: Convolutional Neural Network (CNN)
* **Framework**: TensorFlow / Keras
* **Dataset**: SOCOFing — fingerprint dataset with blood group labels

---

## 📌 Features

* Automated preprocessing of fingerprint images
* Blood group classification using trained ML model
* Easy-to-use prediction script

---

## 📜 License

This project is licensed under the MIT License — you are free to use and modify it.

