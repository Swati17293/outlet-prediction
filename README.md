# Are You Following the Right News-Outlet? A MachineLearning based approach for outlet prediction

This project is an approach to recommend a list of probable outlets covering an event of interest. 

To run this project go through the following steps:

To run the tool, Python 2.7+ must be downloaded. The easiest way to download all necessary Python packages is using pip. To do so, navigate to the project root directory and run:
```
# clone the repository
git clone https://github.com/Swati17293/outlet-prediction.git
cd outlet-prediction
pip install -r requirements.txt
```

### Vectorize the data for future use
```
python3 src/vectorize_data.py
```

### Train the model
```
python3 src/train_model.py
```

### Generate the predictions
```
python3 src/predict_model.py
```

### Evaluate and compare the models
```
python3 src/evaluate_model.py
```

### Pretrained models
Pretrained models are available in /predicted_outlets directory.

### Feature vectors
Feature vectors are available in data/vectorized_data directory.

## License
MIT License