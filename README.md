# NLP Project

This repository contains a full NLP processing pipeline including data loading, preprocessing, model training, and evaluation.

## 📁 Repository Structure
```
.
├── src/
│   ├── main.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   └── utils.py
├── notebooks/
│   └── NLP_Project.ipynb
├── data/
│   └── README.md (instructions for dataset)
├── run.sh
├── requirements.txt
└── LICENSE
```

## 🚀 How to Run

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Run the full pipeline
On Linux / Mac:
```
chmod +x run.sh
./run.sh
```

On Windows:
```
python src/main.py
```

## 🧠 What the Pipeline Does
- Loads dataset  
- Preprocesses text  
- Trains an NLP model  
- Evaluates the model  
- Saves results  

## 📦 Requirements
All dependencies are listed in `requirements.txt`.

## 📜 License
This project is released under the MIT License.  
See the `LICENSE` file for details.
