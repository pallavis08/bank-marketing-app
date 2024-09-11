This ML project is for marketing team in a bank. Objective of this project is to determine if a customer will subscribe for a term deposit.

## Dataset Information

This dataset was downloaded from UCI ML repository (Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306.). This is a multivariate dataset with ~ 45k samples. It has 16 features that consist of both categorical and numerical features. Here is the list of all variables in the dataset:

   ### bank client data:
   1 - age (numeric)
   2 - job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
            "blue-collar","self-employed","retired","technician","services") 
   3 - marital : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)
   4 - education (categorical: "unknown","secondary","primary","tertiary")
   5 - default: has credit in default? (binary: "yes","no")
   6 - balance: average yearly balance, in euros (numeric) 
   7 - housing: has housing loan? (binary: "yes","no")
   8 - loan: has personal loan? (binary: "yes","no")
   ### related with the last contact of the current campaign:
   9 - contact: contact communication type (categorical: "unknown","telephone","cellular") 
  10 - day: last contact day of the month (numeric)
  11 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
  12 - duration: last contact duration, in seconds (numeric)
   ### other attributes:
  13 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
  14 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)
  15 - previous: number of contacts performed before this campaign and for this client (numeric)
  16 - poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")

  Output variable (desired target):
  17 - y - has the client subscribed a term deposit? (binary: "yes","no")

  ## Methodology

  Following steps were used during this project:
   - Exploratory Data Analysis: Code is in EDA.ipynb
   - Model training: A set of model were tried and their performance was evaluated. Code srored in model_training.ipynb file
   - Inference: Best model was saved in model training step and that model is used to predict for any new data. Code is in app.py file. This was done in 2 steps:
    1. First data preparation was done with feature engineering and data cleaning.
    2. Processed data was used for inference
  - Deployment: The model is deployed using Stremlit on their community server. Try it out at: https://bankmarketingapp.streamlit.app/ . You can use a sample csv file file_to_predict.csv (provided).