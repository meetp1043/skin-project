@echo off
echo Setting up project...

python -m venv venv
call venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

mkdir outputs
mkdir outputs\models
mkdir outputs\plots

python split_dataset.py
python eda_preprocessing.py
python models\train_model.py

streamlit run app.py
pause