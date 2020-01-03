import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression

def make_prediction():
    oe = OrdinalEncoder()
    oe2 = OrdinalEncoder()
    train_data=pd.read_csv("Train.csv")
    test_data=pd.read_csv("Test.csv")

    test_data=oe.fit_transform(test_data)

    model = LogisticRegression()
    train_cols=[col for col in train_data.columns if col!="netgain"]
    X= train_data[train_cols]
    X=oe.fit_transform(X).astype("int")


    Y = train_data["netgain"]
    Y = Y.values.reshape(-1,1)
    Y=oe2.fit_transform(Y).astype("int")

    #fitting model with prediction   data and telling it my target
    model.fit(X, Y)

    test_data2 = pd.read_csv("Test.csv")

    test_data2["netgain"]=oe2.inverse_transform(model.predict(test_data).astype("int").reshape(-1,1))
    test_data2[["id", "netgain"]].to_csv("Results.csv")
    # Now we want to delete the first column which is unnecessary since it is jsut for numbering the rows
    new_df = pd.read_csv('Results.csv')
    first_column = new_df.columns[0]
    new_df = new_df.drop([first_column], axis=1)
    new_df.to_csv('Results.csv', index=False)

if __name__=="__main__":
    make_prediction()
