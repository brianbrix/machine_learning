This is a machine learning proigram to predict the Success of an advert
Approach:
To make this prediction:
1. Read the training and test files into variables
2. get the model_features and target features from the  training data. The target column is "netgain"
3. Make the necessary transformations of string values to float/int formats that can be manipulated by the algorithms
4. Make the fitting  using the X, Y columns gotten from step 2
5. Do a prediction using the test data
6. Do an inverse transform on the obtained prediction inorder to get the actual values(original string values)
7. Write the results to Results.csv and do the necessary formatting
Tools:
Pandas- for dataframe manipulation
OrdinalEncoder - for data transformation
LogisticRegression - for making the prediction. We use this model since this is a binary prediction
