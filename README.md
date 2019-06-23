# TimeSeries

The project basically gives a prototypical implementation of carrying out Time series classification using LSTM.
Implemnetation is done to support multi class(4 classes) classification and code utilizes the Keras Library with Tensorflow as backend
Project consists of 3 native source files, with additional files dataset files generated out of the code. 

---------
Contents
---------
    |
    |----- Clean.py -> performs the reading of dataset, arranging, stratified splitting and writing    
    |
    |----- TrainData.py -> performs the LSTM training and saves the trained model 
    |
    |----- TestData.py -> loads the trained model and test it on Testing data computing the metrics
    |
    |----- Train_data.csv -> Training data series
    |----- Train_label.csv -> Training data labels
    |
    |----- Valid_data.csv -> Validation data series
    |----- Valid_label.csv -> Validation data labels
    |
    |----- Test_data.csv -> Testing data series
    |----- Test_label.csv -> Testing data labels
    |
    |----- TimeSeries_LSTM.h5 -> Trained model saved in h5 format
    |
    |----- Readme -> helper text file
        
--------
 Usage
--------

- Data Cleaning and splitting -> python Dataclean.py
- Training                    -> python TrainData.py
- Testing                     -> python TestData.py
        
        
