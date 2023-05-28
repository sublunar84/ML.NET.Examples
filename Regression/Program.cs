using Microsoft.ML;
using Microsoft.ML.Data;

var projectDirectory = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../"));
var assetsRelativePath = Path.Combine(projectDirectory, "assets");
var workspaceRelativePath = Path.Combine(projectDirectory, "workspace");

Console.WriteLine("Press A to train and save a model, press B to make predicitions with a saved model");

var answer = Console.ReadKey();

switch (answer.KeyChar.ToString().ToUpper())
{
    case "A":
        TrainModel();
        break;
    case "B":
        MakePredictions();
        break;
    default:
        Console.WriteLine("Wrong answer!");
        break;

}

void TrainModel()
{
    MLContext mlContext = new MLContext();

    IDataView dataView = mlContext.Data.LoadFromTextFile<ModelInput>(Path.Combine(assetsRelativePath, "Life Expectancy Data.csv"), hasHeader: true, separatorChar: ',');

   var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "LifeExpectancy")
       .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "StatusEncoded", inputColumnName: "Status"))
       .Append(mlContext.Transforms.Concatenate("Features", "StatusEncoded", "AdultMortality", "InfantDeaths", "Alcohol", "PercentageExpenditure", "HepatitisB", "Measles", "BMI"))
       .Append(mlContext.Regression.Trainers.FastTree());

    // Shuffle rows
    dataView = mlContext.Data.ShuffleRows(dataView);

    // split into test and train sets
    var trainSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
    var validationTestSplit = mlContext.Data.TrainTestSplit(trainSplit.TestSet);

    IDataView trainSet = trainSplit.TrainSet;
    IDataView validationSet = validationTestSplit.TrainSet;
    IDataView testSet = validationTestSplit.TestSet;

    var trainedModel = pipeline.Fit(trainSet);

    SaveModel(mlContext, trainedModel, dataView);
    Evaluate(mlContext, trainedModel, testSet);
    MakePredictions(validationSet);
}

void SaveModel(MLContext mlContext, ITransformer trainedModel, IDataView data)
{
    // Save Trained Model
    mlContext.Model.Save(trainedModel, data.Schema, Path.Combine(workspaceRelativePath, "model.zip"));
}

void Evaluate(MLContext mlContext, ITransformer model, IDataView testData)
{
    //IDataView dataView = mlContext.Data.LoadFromTextFile<ModelInput>(Path.Combine(assetsRelativePath, "taxi-fare-test.csv"), hasHeader: true, separatorChar: ',');
    var predictions = model.Transform(testData);
    var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

    Console.WriteLine();
    Console.WriteLine($"*************************************************");
    Console.WriteLine($"*       Model quality metrics evaluation         ");
    Console.WriteLine($"*------------------------------------------------");
    Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");
    Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:#.##}");

}

void MakePredictions(IDataView validationSet = null)
{
    MLContext mlContext = new MLContext();

    // Load Trained Model
    ITransformer trainedModel = mlContext.Model.Load(Path.Combine(workspaceRelativePath, "model.zip"), out var modelSchema);

    TestSinglePrediction(mlContext, trainedModel, validationSet);

}

void TestSinglePrediction(MLContext mlContext, ITransformer model, IDataView validationSet)
{
    var predictionFunction = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);

    // If validationSet is null, use random rows from file Life Expectancy Data.csv:
    if (validationSet == null)
    {
        var dataView = mlContext.Data.LoadFromTextFile<ModelInput>(Path.Combine(assetsRelativePath, "Life Expectancy Data.csv"), hasHeader: true, separatorChar: ',');
        validationSet = mlContext.Data.ShuffleRows(dataView);
        validationSet = mlContext.Data.TakeRows(dataView, 3);
    } 

    mlContext.Data.CreateEnumerable<ModelInput>(validationSet, reuseRowObject: false).Take(3).ToList().ForEach(sample =>
    {
        var prediction = predictionFunction.Predict(sample);

        Console.WriteLine($"**********************************************************************");
        Console.WriteLine($"Predicted life expectancy: {prediction.LifeExpectancy:0.####}, actual life expectancy: {sample.LifeExpectancy}");
        Console.WriteLine($"**********************************************************************");
    });
    

}
public class ModelInput
{
    // Country,Year,Status,Life expectancy ,Adult Mortality,infant deaths,Alcohol,percentage expenditure,Hepatitis B,Measles , BMI ,under-five deaths ,Polio,Total expenditure,Diphtheria , HIV/AIDS,GDP,Population, thinness  1-19 years, thinness 5-9 years,Income composition of resources,Schooling

    [LoadColumn(2)]
    public string? Status;

    [LoadColumn(3)]
    public float LifeExpectancy;

    [LoadColumn(4)]
    public float AdultMortality;

    [LoadColumn(5)]
    public float InfantDeaths;

    [LoadColumn(6)]
    public float Alcohol;

    [LoadColumn(7)]
    public float PercentageExpenditure;

    [LoadColumn(8)]
    public float HepatitisB;

    [LoadColumn(9)]
    public float Measles;

    [LoadColumn(10)]
    public float BMI;

}

public class ModelOutput
{
    [ColumnName("Score")]
    public float LifeExpectancy;
}
