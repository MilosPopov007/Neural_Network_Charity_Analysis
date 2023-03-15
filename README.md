# Neural_Network_Charity_Analysis
Neural Networks and Deep Learning Models


Nonprofit foundation "Alphabet Soup" is a a philanthropic foundation dedicated to helping organizations that protect the environment and prove people's well-being and unify the world. "Alphabet Soup" has raised and donated over $10 billion in the past 20 years.<br> This money has been used to invest in a life saving technologies and organized reforestation groups around the world. Unfortunately, not every donation the company makes is impactful.<br> In some cases, an organization will take the money and  disappear as a result.<br><br> "Alphabet soups" President Andy Glad has asked me and my team to predict which organizations are worth donating to and which are too high risk. He wants us to create a mathematical, data-driven solution that can do this accurately.<br> We decided that this problem is too complex for the statistical, machine learning. <br>Instead we will design and train a Deep Learning Neural Network.<br> This model will evaluate all types of input data and produce a clear decision making result.<br> A neural network is a powerful machine learning technique that is modeled after neurons in the brain.<br> Neural networks can rival the performance of the most robust statistical algorithms without having to worry about any statistical theory.<br> In this analysis, we will train, evaluate, and export neural network models to use in any scenario with the help of the Python TensorFlow Library.



## Results:

The main purpose of Neural_Network_Charity_Analysis is to create a binary classifier that is capable of predicting whether applicants will be successful if funded by "Alphabet Soup". <br><br>
From "Alphabet Soup’s" business team, we received a CSV containing more than 34,000 organizations that have received funding from "Alphabet Soup" over the years.<br> Within this dataset are a number of columns that capture metadata about each organization, such as the following:

* EIN and NAME — Identification columns
* APPLICATION_TYPE — Alphabet Soup application type
* AFFILIATION — Affiliated sector of industry
* CLASSIFICATION — Government organization classification
* USE_CASE — Use case for funding
* ORGANIZATION — Organization type
* STATUS — Active status
* INCOME_AMT — Income classification
* SPECIAL_CONSIDERATIONS — Special consideration for application
* ASK_AMT — Funding amount requested
* IS_SUCCESSFUL — Was the money used effectively

The target variable for our model is the "IS_SUCCESSFUL" column, which indicates whether a charity donation was used effectively.<br>
The features for our model include all other columns in the dataset, except for the "EIN" and "NAME" columns. These columns do not provide any useful information for predicting the success of a charity donation.<br>
The "EIN" and "NAME" columns are neither targets nor features, and should be removed from the input data. We dropped these columns using the Pandas DataFrame method drop().<br><br>
Breakdown of the steps in Neural_Network_Charity_Analysis:<br>

### Preprocessing Data for a Neural Network Model:

* Load and transform the input data
* Preprocess the data by performing the following steps :
* Drop the non-beneficial ID columns
* Encode the categorical variables using one-hot encoding
* Standardize the numerical variables using the StandardScaler() function
* Split the preprocessed data into features and target arrays
* Split the preprocessed data into training and testing datasets
* Save the trained model to an HDF5 file

### Compile, Train, and Evaluate the Model:

* Define the model architecture using the Sequential() function from the TensorFlow Keras library
* Add layers to the model using the Dense() function from the TensorFlow Keras library
* Compile the model using binary_crossentropy loss function, adam optimizer, and accuracy metric
* Train the model using the fit() function and the training dataset
* Evaluate the model using the testing dataset and print the loss and accuracy scores

### Optimize the Model:
