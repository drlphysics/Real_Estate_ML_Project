# AI-Driven Real Estate Market Insights
#### Utilizing AI and machine learning, we analyze and predict the dynamics of the real estate market in parts of North Carolina using data from the MLS.

# Table of Contents 

## 1. [Methodology](#methodology)
- [Data Collection](#data-collection)
- [Data Exploration](#data-exploration)
- [Data Preprocessing](#data-preprocessing)
- [Model Selection](#model-selection)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
## 2. [Tools](#tools)
## 3. [Team Members](#team-members)
## 4. [Installation](#installation)
- [Steps to install](#steps-to-install)
- [Dependencies](#dependencies)
## 5. [Usage](#usage)

## 6. [Contribution](#contribution)

## Methodology <a name="methodology"></a>

### Data Collection <a name="data-collection"></a>
 - Sources: MLS Data for the Triangle Area
 
 - Data Features: MLS #, Class, Property Type, Address, City, Zip, Neighborhood, Subdivision, Bedrooms, Total Baths, Total Living Area SqFt, Acres, Year Built, List Date, Closing Date, Days On Market, List Price, Sold Price, and Unna




### Data Exploration <a name="data-exploration"></a> 
  - Visualization: Matplotlib, pandas.plotting, seaborn 

  - Statistical Analysis: '.describe()', '.info()', '.value_counts()', 'z-score', 'df.corr()' 


### Data Preprocessing <a name="data-preprocessing"></a> 
 - Data Cleaning: Handling null entries, correcting errors, dropping missing values, converting value types to integers. 


 - Transformation: 'utils.scale_data'
    - 'StandardScaler()'


### Model Selection <a name="model-selection"></a> 
 - Ridge Regression
 - Lasso Regression
 - Random Forest
 - Decision Tree
 - SVM
 - KNN

### Model Training <a name="model-training"></a> 
 - Data Splits: 'utils.split_data'
    - 'train_test_split()'


- Hyperparameter Tuning: Data Models Optimization will be in the 'mls_utils File'--> 'utils.tune_hyperparameters'



### Model Evaluation <a name="model-evaluation"></a> 
- Scoring Methods: 'utils.evaluate_model'
    - mean_squared_error MSE 
    - r2 score 



### Tools <a name="tools"></a>
- Programming Language: Python
- Libraries: Pandas, NumPy, Scikit-learn, datetime
- Visualization Tools: Matplotlib, Seaborn
- Utils File: 'mls_utils.py'
    - splits data
    - scales data
    - initiates models
    - tunes hyperparameters
    - calculates bias variance
    - evaluates models


### Team Members <a name="team-members"></a>
- Richard Lankford 
    - Github: https://github.com/rwlankford
    - Linkedin: https://www.linkedin.com/in/rwlankford/
- Amelia Fernandez
    - Github: https://github.com/aFernandez88
    - Linkedin: https://www.linkedin.com/in/amelia-fernandez-17a56220b/
- David Little
    - Github: https://github.com/drlphysics
    - Linkedin: https://www.linkedin.com/in/david-little-phd-67b360185/

### Installation <a name="installation"></a>
- ### Steps to install <a name="steps-to-install"></a>
    1. Download the latest version of Python
    2. pip install matplotlib scipy numpy Pandas scikit-learn seaborn jupyter mls_utils
    3. Use the 'git clone' command to clone repository: https://github.com/drlphysics/Real_Estate_ML_Project.git
    4. 'cd' Real_Estate_ML_Project
- ### Dependencies <a name="dependencies"></a>
- Python 3.8+
- Pandas
- Numpy
- Scikit-learn
- Matplotlib
- Seaborn
- scipy.stats
- Jupyter
- mls_utils

## Usage <a name="usage"></a>
  1. Explore the data
  2. Preprocess the data
  3. Train the model
  4. Evaluate the model
  5. Deploy the model


## Contribution <a name="contribution"></a>
Your contributions are highly valued! Feel free to submit a pull request or open an issue to share your ideas





