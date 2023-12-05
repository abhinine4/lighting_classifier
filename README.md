# Lighting Classifier

A Random Forest model that parses pdf documents, classifies and predicts whether the pdf is for a lighting product or not.

The RF model achieves **88.75%** on test dataset.

### Setup
- Create conda environment

    ```conda create --name test_par python=3.10```

- Activate environment

    ```conda activate test_par```

- Install the required libraries 

    ```pip install -r requirements.txt```

### Usage
- Get results (URL or local pdf file)

     ```python predict.py --filepath “URL or local path to file” ```

- To train text classifier, run :

    ```python train.py ```

- Run evaluation on Test data : 

    ```python eval.py```

- Scrape and preprocess train and test data from csv files :

    Download and process text in pdf files from csv files.

    ```
    python ./data/data_scraping.py
    python extract_text.py
    ```