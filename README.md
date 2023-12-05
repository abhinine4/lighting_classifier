# Lighting Classifier

A Random Forest model that parses pdf documents, classifies and predicts whether the pdf is for a lighting product or not.

The RF model achieves **88.75%** accuracy on test dataset.

Below is the confusion matrix for test dataset :

![CM](test_confusion_matrix.png)

### Setup
- Create conda environment :

    ```bash
    conda create --name parspec python=3.10
    ```

- Activate environment :

    ```bash
    conda activate parspec
    ```

- Install the required libraries :

    ```bash
    pip install -r requirements.txt
    ```

### Usage
- Get results (URL or local pdf file) :

     ```
     python predict.py --filepath “URL or local path to file” 
     
     ```

- Run only evaluation on test data : 

    ```
    python eval.py
    ```

### Training

-  Dataset :

    Download pdf files and process text :

    ```bash
    ./data_processor.sh
    ```

- Train text classifier :

    ```
    python train.py 
    ```



## NOTE

Use "lines=True" to read processed JSON files with pandas.read_json()

