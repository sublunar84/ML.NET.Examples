Simple .NET 6 console app to test AutoML and Text multiclassification with ModelBuilder in VS

To use:
* In assets - put the training data (I used this set from kaggle on luxury products apparel: https://www.kaggle.com/datasets/chitwanmanchanda/luxury-apparel-data)
* Install Visual studio ML.NET ModelBuilder extension
* Right click on the project and select "Add" -> "Machine Learning Model"
* Do a Data classification task using the data i assets. A new file "TextClassification.mbconfig" will be generated including the model, the code to evaluate it and the code to use it
* Make a prediction by running the app
