# Grey Christmas

A machine learning program that attempts to predict whether a given year will have a snowy Christmas based on the first 11 months of the year

# Demo

This demo shows the model being trained on 131 years of data from the Vancouver region. The test dataset used is the same dataset to demonstrate the model's ability to converge, or at least memorize data.


https://github.com/user-attachments/assets/aa2407fa-c40a-4897-8cd3-e3179d2794bd


# Usage

The `model` module can be imported. It provides 2 classes: `Trainer` and `Predictor`. As the names suggest, `Trainer` is used to train a fresh model from scratch, whereas `Predictor` loads a model from a file and given the data from a year, predicts the probability of snow.

`Trainer` takes in two lists of filepaths during initialization, one for training and the second for testing. Currently, only the format used by the Canadian government's daily climate observation dataset is accepted. Then, by using the `.train` method, you can specify the number of epochs the model attempts to learn from the dataset. Using the `.save` method, the class will export the trained model as a pytorch weights file.

`Predictor` takes in a path to a pytorch weights file, then initializes and loads the model. Then, the `.predict` or `.predict_tensor` methods, it will take the data from either a pandas dataframe or a pytorch tensor to predict whether or not the year will end up snowing during Christmas.

# Design

The `Forecast` model behind everything takes heavy inspiration from image classification models like ResNet. It uses similar blocks to it, but visualizes a year as a 6-channel vector with temperature and precipitation information. Then, it uses 1-dimensional convolutions to attempt to classify the vector as a year that will lead to a snowy or unsnowy Christmas.

The data used is from (climatedata.ca)[climatedata.ca], and consequently the weather data class knows only how to accept data from `.csv` files that contain all the headers `CLIMATE_IDENTIFIER`, `LOCAL_YEAR`, `LOCAL_MONTH`, `LOCAL_DAY`, `MEAN_TEMPERATURE`, `MIN_TEMPERATURE`, `MAX_TEMPERATURE`, `TOTAL_PRECIPITATION`, `TOTAL_RAIN`, and `TOTAL_SNOW`.
