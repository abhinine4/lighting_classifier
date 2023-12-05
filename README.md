# Lighting Classifier

A Random Forest model that parses pdf documents, classifies and predicts whether the pdf is for a lighting product or not. 

### Usage
- Create conda environment
    ```conda create --name test_par python=3.10```

- Activate environment
    ```conda activate test_par```

- Install the required libraries 
    ```pip install -r requirements.txt```

- Run the script 
     ```python predict.py --filepath “URL or local path to file” ```

- Train text classifier
    ```python train.py ```

- Run evaluation on Test data
    ```python eval.py```
