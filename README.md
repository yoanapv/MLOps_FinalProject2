# MLOps_FinalProject
This is a repo to Final project

**Subject:** MLOps
**Teacher:** 
**Student:** A01688744 - Norma Yoana Pérez Villalva

**HOTEL RESERVATIONS**

**Problem type:** Classification (Binary)

**Problem description:**
*   The online hotel reservation channels have dramatically changed booking possibilities and customers’ behavior
*   A significant number of hotel reservations are called-off due to cancellations or no-shows.
*   The typical reasons for cancellations include change of plans, scheduling conflicts, etc. This is often made easier by the option to do so free of charge or preferably at a low cost which is beneficial to hotel guests but it is a less desirable and possibly revenue-diminishing factor for hotels to deal with.

**Goal:**
We need to determine if each customer is likely to cancel their reservation or not.

**Source:**
     Problem: https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset

     Notbook selected: https://www.kaggle.com/code/raphaelmarconato/hotel-reservations-eda-balancing-and-ml-93-4



**BASELINE MODEL**
5. To review the baseline model go to the Baseline folder and check the README_BASELINE file, or click in the next link: https://github.com/yoanapv/MLOps_FinalProject2/blob/main/Baseline/README_BASELINE.md


### Run the existing notebook
1. Clone the project `https://github.com/yoanapv/MLOps_FinalProject2.git` on your local computer.
2. Open the Terminal
    * Clic on Terminal --> New Terminal

    * Press Ctrl + Shift + Ñ

3. Create a virtual environment with `Python 3.10.9`
    * Create venv (Windows)
        ```
        py -3.10 -m venv venv
        ```

    * Create venv (mac)
        ```
        python3 -m venv venv
        ```

    * Activate the virtual environment (Windows)

        ```
        venv/scripts/Activate.ps1
        ```

    * Activate the virtual environment (mac)

        ```
        source venv/bin/activate
        ```
        You'll see the name of the virtual environment, in parentheses, at the beginning of the code line.

4. Install libraries
    Run the following command to install the libraries.

    ```bash
    pip install -r requirements.txt
    ```

## Model training from a main file

To train the Logistic Model, only run the following code:

```bash
python mlops_finalproject/mlops_finalproject.py
```

Output:

```bash
test roc-auc : 0.8528526687921574
test accuracy: 0.7456926257753274
Model saved in ./models/extra_trees_classifier_model_output.pkl
```

## Execution of unit tests (Pytest)

### Test location

You can find the test location in the [test] folder, and the following tests:

* Test `test_csv_file_existence`:  
Test case to check if the CSV file exists.

* Test `test_categorical_imputer`:  
Test the `transform` method of the CategoricalImputer transformer.

* Test `test_ordering_features`:  
Test the `transform` method of the OrderingFeatures transformer.

* Test `test_model_and_pipeline_saved`:  
Test to validate the existence of a `.pkl` model and pipeline files.

### Execution instructions

#### Test `Data Retriever` class

The following test validates the [load_data.py](itesm_mlops_project/load/load_data.py) module, with the `DataRetriever` class.

Follow the next steps to run the test.

* Run in the terminal:

    ```bash
    pytest ./tests/test_mlops_finalproject.py::test_csv_file_existence -v
    ```

* You should see the following data output:

    ```pytest
    ============================================================= test session starts ==============================================================
    platform win32 -- Python 3.10.9, pytest-7.4.0, pluggy-0.13.1 -- C:\Users\norma.perez\Documents\GitHub\MLOps_FinalProject\venv3_10\Scripts\python.exe
    cachedir: .pytest_cache
    rootdir: C:\Users\norma.perez\Documents\GitHub\MLOps_FinalProject2\mlops_finalproject
    plugins: anyio-3.7.1
    collected 1 item

    tests/test_mlops_finalproject.py::test_csv_file_existence PASSED                                                                          [100%]

    ============================================================== 1 passed in 19.33s ============================================================== 
    ```

#### Test `MissingIndicator` class - `transform` method

The following test validates the [preprocess_data.py](mlops_finalproject/preprocess/preprocess_data.py) module, with the `CategoricalImputer` class in the `transform` method.

Follow the next steps to run the test.

* Run in the terminal:

    ```bash
    pytest ./tests/test_mlops_finalproject.py::test_categorical_imputer -v
    ```

* You should see the following data output:

    ```pytest
    ======================================================================== test session starts ========================================================================
    platform win32 -- Python 3.10.9, pytest-7.4.0, pluggy-0.13.1 -- C:\Users\norma.perez\Documents\GitHub\MLOps_FinalProject\venv3_10\Scripts\python.exe
    cachedir: .pytest_cache
    rootdir: C:\Users\norma.perez\Documents\GitHub\MLOps_FinalProject2\mlops_finalproject
    plugins: anyio-3.7.1
    collected 1 item

    tests/test_mlops_finalproject.py::test_categorical_imputer PASSED                                                                                              [100%] 

    ========================================================================= 1 passed in 5.66s =========================================================================    
    ```
#### Test `test_ordering_features` class - `fit` method

The following test validates the [preprocess_data.py](itesm_mlops_project/preprocess/preprocess_data.py) module, with the `OrderingFeatures` class in the `transform` method.

Follow the next steps to run the test.

* Run in the terminal:

    ```bash
    pytest ./tests/test_mlops_finalproject.py::test_ordering_features -v
    ```

* You should see the following data output:

    ```pytest
    =========================================== test session starts ===========================================
    platform win32 -- Python 3.10.9, pytest-7.4.0, pluggy-0.13.1 -- C:\Users\norma.perez\Documents\GitHub\MLOps_FinalProject\venv3_10\Scripts\python.exe
    cachedir: .pytest_cache
    rootdir: C:\Users\norma.perez\Documents\GitHub\MLOps_FinalProject2\mlops_finalproject
    plugins: anyio-3.7.1
    collected 1 item

    tests/test_mlops_finalproject.py::test_ordering_features PASSED                                 [100%]

    ============================================ 1 passed in 5.92s ============================================
    ```
#### Test model existence

The following test validates the model's existence after the training.

Follow the next steps to run the test.

* Run in the terminal:

    ```bash
    pytest ./tests/test_mlops_finalproject.py::test_model_and_pipeline_saved -v -s
    ```

* You should see the following data output:

    ```pytest
    =========================================== test session starts ===========================================
    platform win32 -- Python 3.10.9, pytest-7.4.0, pluggy-0.13.1 -- C:\Users\norma.perez\Documents\GitHub\MLOps_FinalProject\venv3_10\Scripts\python.exe
    cachedir: .pytest_cache
    rootdir: C:\Users\norma.perez\Documents\GitHub\MLOps_FinalProject2\mlops_finalproject
    plugins: anyio-3.7.1
    collected 1 item

    tests/test_mlops_finalproject.py::test_model_and_pipeline_saved PASSED

    ============================================ 1 passed in 5.78s ============================================
    ```
## Usage

### Individual Fastapi and Use Deployment

* Run the next command to start the Titanic API locally

    ```bash
    uvicorn itesm_mlops_project.api.main:app --reload
    ```

#### Checking endpoints

1. Access `http://127.0.0.1:8000/`, you will see a message like this `"HotelReservation classifier is all ready to go!"`
2. Access `http://127.0.0.1:8000/docs`, the browser will display something like this:
![FastAPI Docs](docs/imgs/fast-api-docs.png)
3. Try running the following predictions with the endpoint by writing the following values:
    * **Prediction**  
        Request body

        ```bash
        {
        "no_of_week_nights": 10,
        "lead_time": 50,
        "arrival_month": 10,
        "arrival_date": 10,
        "avg_price_per_room": 350.0,
        "no_of_special_requests": 0
        }
        ```

        Response body
        The output will be:

        ```bash
        "Resultado predicción: [0]"
        ```
### Individual deployment of the API with Docker and usage

#### Build the image

* Ensure you are in the `itesm_finalproject/` directory (root folder).
* Run the following code to build the image:

    ```bash
    docker build -t hotelreservations-image ./mlops_finalproject/app/
    ```

* Inspect the image created by running this command:

    ```bash
    docker images
    ```

    Output:

    ```bash
    REPOSITORY                TAG       IMAGE ID       CREATED          SIZE     
    hotelreservations-image   latest    f29e6f736257   23 seconds ago   609MB
    ```
#### Run Hotel Reservations REST API

1. Run the next command to start the `hotelreservations-image` image in a container.

    ```bash
    docker run -d --rm --name hotelreservations-c -p 8000:8000 hotelreservations-image
    ```

2. Check the container running.

    ```bash
    docker ps -a
    ```

    Output:

    ```bash
   CONTAINER ID   IMAGE                            COMMAND                  CREATED          STATUS                     PORTS     NAMES
    01db05c16368   hotelreservations-image:latest   "uvicorn main:app --…"   11 seconds ago   Exited (1) 8 seconds ago             affectionate_wescoff
    ```
#### Checking endpoints for app

1. Access `http://127.0.0.1:8000/`, and you will see a message like this `"Hotel Reservations classifier is all ready to go!"`
2. A file called `main_api.log` will be created automatically inside the container. We will inspect it below.
3. Access `http://127.0.0.1:8000/docs`, the browser will display something like this:
    ![FastAPI Docs](docs/imgs/fast-api-docs.png)

4. Try running the following predictions with the endpoint by writing the following values:
    * **Prediction **  
        Request body

        ```bash
        {
        "no_of_week_nights": 10,
        "lead_time": 60,
        "arrival_month": 12,
        "arrival_date": 15,
        "avg_price_per_room": 800.0,
        "no_of_special_requests": 0
        }
        ```

        Response body
        The output will be:

        ```bash
        "Resultado predicción: [1]"
        ```
#### Opening the logs in Frontend

Open a new terminal, and execute the following commands:

1. Copy the `frontend` logs to the root folder:

    ```bash
    docker cp mlops_finalproject:/frontend.log .
    ```


2. You can inspect the logs and see something similar to this:

    ```bash
    INFO: 2023-08-21 23:42:00,057|main|Front-end is all ready to go!
    INFO: 2023-08-21 23:45:04,575|main|Front-end is all ready to go!
    DEBUG: 2023-08-21 23:45:43,724|main|Incoming input in the front end: {'pclass_nan': 0, 'age_nan': 0, 'sibsp_nan': 0, 'parch_nan': 0, 'fare_nan': 0, 'sex_male': 1, 'cabin_Missing': 1, 'cabin_rare': 0, 'embarked_Q': 1, 'embarked_S': 0, 'title_Mr': 1, 'title_Mrs': 0, 'title_rar': 0}
    DEBUG: 2023-08-21 23:45:43,742|main|Prediction: "Resultado predicción: [0]"
    DEBUG: 2023-08-21 23:46:47,024|main|Incoming input in the front end: {'pclass_nan': 0, 'age_nan': 0, 'sibsp_nan': 1, 'parch_nan': 0, 'fare_nan': 0, 'sex_male': 0, 'cabin_Missing': 0, 'cabin_rare': 0, 'embarked_Q': 1, 'embarked_S': 0, 'title_Mr': 1, 'title_Mrs': 0, 'title_rar': 0}
    DEBUG: 2023-08-21 23:46:47,038|main|Prediction: "Resultado predicción: [1]"
    ```

