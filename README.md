# Neural_Network_Charity_Analysis
Neural Networks and Deep Learning Models


Nonprofit foundation "Alphabet Soup" is a a philanthropic foundation dedicated to helping organizations that protect the environment, prove people's well-being and unify the world. "Alphabet Soup" has raised and donated over $10 billion in the past 20 years.<br> This money has been used to invest in a life saving technologies and organized reforestation groups around the world. Unfortunately, not every donation the company makes is impactful.<br> In some cases, an organization will take the money and  disappear as a result.<br><br> "Alphabet soups" President Andy Glad has asked me and my team to predict which organizations are worth donating to and which are too high risk. He wants us to create a mathematical, data-driven solution that can do this accurately.<br> We decided that this problem is too complex for the statistical, machine learning. <br>Instead we will design and train a Deep Learning Neural Network.<br> This model will evaluate all types of input data and produce a clear decision making result.<br> A neural network is a powerful machine learning technique that is modeled after neurons in the brain.<br> Neural networks can rival the performance of the most robust statistical algorithms without having to worry about any statistical theory.<br> In this analysis, we will train, evaluate, and export neural network models to use in any scenario with the help of the Python TensorFlow Library.<br><br>
![This is an image](https://github.com/MilosPopov007/Neural_Network_Charity_Analysis/blob/main/Resources/DeepLearning.jpg)



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
Breakdown of the steps in [Neural_Network_Charity_Analysis](https://github.com/MilosPopov007/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity.ipynb):<br>

### Preprocessing Data for a Neural Network Model:

* Load and transform the input data
* Preprocess the data by performing the following steps :
* Drop the non-beneficial ID columns
* Encode the categorical variables using one-hot encoding
* Standardize the numerical variables using the StandardScaler() function
* Split the preprocessed data into features and target arrays
* Split the preprocessed data into training and testing datasets

### Compile, Train, and Evaluate the Model:

* Define the model architecture using the Sequential() function from the TensorFlow Keras library
* Add layers to the model using the Dense() function from the TensorFlow Keras library
* Compile the model using binary_crossentropy loss function, adam optimizer, and accuracy metric
* Train the model using the fit() function and the training dataset
* Evaluate the model using the testing dataset and print the loss and accuracy scores
* Create a checkpoint system to save the model weights every 5 epochs
* Save the trained model to an HDF5 file<br>

For our neural network model, we selected a sequential model with three dense layers.<br> The first layer had 80 neurons, the second layer had 30 neurons, and the third layer had one neuron.<br> We chose these numbers based on some trial and error, trying to find a balance between complexity and simplicity.<br> We used the ReLU activation function for the first two layers and a sigmoid activation function for the output layer.<br><br>
![This is an image](https://github.com/MilosPopov007/Neural_Network_Charity_Analysis/blob/main/Resources/model_base.png)

During training, we were able to achieve an accuracy of 72.56% and a loss of 0.5744.<br> While this was close to our target performance of 75% accuracy, we still wanted to try and optimize the model further to see if we could improve performance.<br><br>
![This is an image](https://github.com/MilosPopov007/Neural_Network_Charity_Analysis/blob/main/Resources/base_result.png)

### [Optimize the Model](https://github.com/MilosPopov007/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity_Optimization.ipynb):

Using TensorFlow, I tried to optimize our model in order to achieve a target predictive accuracy higher than 75%.<br><br>
#### Model_Attempt_1 

 In our first attempt, I made the following changes to the original model:

 I changed the binning for the application_type feature from original to Less than 1000.
 This allowed us to better capture the patterns in the data and improve the accuracy of the model.

 Decreasing the number of epochs to 50 can potentially reduce overfitting, as the model is less likely to memorize the training data and more likely to learn generalized patterns. It can also speed up the training process, as the model has to process fewer iterations.

 Finally, I added a callback function to save the model weights every 5 epochs.
 This ensured that we could access the best model weights in case the training was interrupted or if we wanted to load the model at a later time.<br><br>
 
 ![This is an image](https://github.com/MilosPopov007/Neural_Network_Charity_Analysis/blob/main/Resources/model_1.png)
 
#### Model_Attempt_2 

 In attempt number 2, I made two changes to the neural network architecture.

 First, I increased the number of neurons in the first and second hidden layers from 80 and 30 to 100 and 50 respectively.
 Second, I added a third hidden layer with 20 neurons.
 We trained the model for 50 epochs and saved the weights every 5 epochs using a callback.<br><br>
 
 ![This is an image](https://github.com/MilosPopov007/Neural_Network_Charity_Analysis/blob/main/Resources/model_2.png)
 
 #### Model_Attempt_3 

I increased the number of neurons in each layer, added a fourth hidden layer, and changed the activation functions of all layers to "selu" except for the last layer which remains "sigmoid".<br>SELU (Scaled Exponential Linear Units) is an activation function commonly used in deep neural networks [^1] .
[^1]:SELU activation function official TensorFlow repository: https://github.com/tensorflow/tensorflow
It is a self-normalizing activation function, which means it maintains a mean activation close to 0 and standard deviation close to 1 during training. This can lead to improved performance and faster convergence in deep neural networks.<br>

The number of epochs is still set to 50 and we are using the same checkpoint callback to save the model weights every 5 epochs.<br><br>

![This is an image](https://github.com/MilosPopov007/Neural_Network_Charity_Analysis/blob/main/Resources/mode_3.png)

#### Model_Attempt_4 

Noisy variables are removed from features ('SPECIAL_CONSIDERATIONS').

To optimize the model, I increased the number of neurons in the first hidden layer to 100, added a third hidden layer with 20 neurons and trained the model for 100 epochs.

I decided to drop "SPECIAL_CONSIDERATIONS" column, because it had very few unique values, and most of the values were "N" with a few "Y"s, which would not contribute much to the model's predictive power.<br>
By dropping this column, I simplified the model and reduced the number of input features.<br><br>

![This is an image](https://github.com/MilosPopov007/Neural_Network_Charity_Analysis/blob/main/Resources/model_4.png)<br><br>

## Summary: 

Based on the results of our attempts to optimize the model, we were not able to achieve a significant improvement in accuracy beyond the base model's performance of 72.56%. While we did see some fluctuation in accuracy with different optimization attempts, these changes were not substantial enough to meet our target performance of 75% or higher.

I recommend considering the use of a different model to solve the classification problem. This could involve exploring different types of neural networks or considering other machine learning algorithms, such as decision trees or support vector machines (SVMs).

The choice of model ultimately depends on the specific characteristics of the dataset and the nature of the classification problem.<br> Exploring alternative approaches may provide a path towards achieving the desired level of performance.


