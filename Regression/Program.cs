using Microsoft.ML;
using Microsoft.ML.Data;

var projectDirectory = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../"));
var assetsRelativePath = Path.Combine(projectDirectory, "assets");
var workspaceRelativePath = Path.Combine(projectDirectory, "workspace");

Console.WriteLine("Press A to train and save a model, press B to make predictions with a saved model");

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

   var pipeline = mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "StatusEncoded", inputColumnName: "Status")
       .Append(mlContext.Transforms.Concatenate("Features", "StatusEncoded", "AdultMortality", "InfantDeaths", "Alcohol", "PercentageExpenditure", "HepatitisB", "Measles", "BMI"))
       .Append(mlContext.Regression.Trainers.FastTree());
    
    // Split into test and train sets
    var trainSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

    IDataView trainSet = trainSplit.TrainSet;
    IDataView testSet = trainSplit.TestSet;

    var trainedModel = pipeline.Fit(trainSet);

    SaveModel(mlContext, trainedModel, dataView);
    Evaluate(mlContext, trainedModel, testSet);
    MakePredictions(testSet);
}

void SaveModel(MLContext mlContext, ITransformer trainedModel, IDataView data)
{
    // Save Trained Model
    mlContext.Model.Save(trainedModel, data.Schema, Path.Combine(workspaceRelativePath, "model.zip"));
}

void Evaluate(MLContext mlContext, ITransformer model, IDataView testData)
{
    var predictions = model.Transform(testData);
    var metrics = mlContext.Regression.Evaluate(predictions);

    Console.WriteLine();
    Console.WriteLine($"*************************************************");
    Console.WriteLine($"*       Model quality metrics evaluation         ");
    Console.WriteLine($"*------------------------------------------------");
    Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");
    Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:#.##}");

}

void MakePredictions(IDataView? testSet = null)
{
    MLContext mlContext = new MLContext();

    // Load Trained Model
    ITransformer model = mlContext.Model.Load(Path.Combine(workspaceRelativePath, "model.zip"), out _);

	var predictionFunction = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);

	// If testSet is null, use random rows from file Life Expectancy Data.csv:
	if (testSet == null)
	{
		var dataView = mlContext.Data.LoadFromTextFile<ModelInput>(Path.Combine(assetsRelativePath, "Life Expectancy Data.csv"), hasHeader: true, separatorChar: ',');
		testSet = mlContext.Data.ShuffleRows(dataView);
		testSet = mlContext.Data.TakeRows(testSet, 3);
	}

	mlContext.Data.CreateEnumerable<ModelInput>(testSet, reuseRowObject: false).Take(3).ToList().ForEach(sample =>
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

    [LoadColumn(3), ColumnName("Label")]
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
