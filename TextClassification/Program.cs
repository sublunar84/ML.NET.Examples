using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;

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
	var mlContext = new MLContext();

	var dataView = mlContext.Data.LoadFromTextFile<ModelInput>(Path.Combine(assetsRelativePath, "Luxury_Products_Apparel_Data.csv"), hasHeader: true, separatorChar: ',');

	var pipeline = mlContext.Transforms.Text.FeaturizeText(inputColumnName: @"ProductName", outputColumnName: @"ProductName")
		.Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: @"Description", outputColumnName: @"Description"))
		.Append(mlContext.Transforms.Concatenate(@"Features", new[] { @"ProductName", @"Description" }))
		.Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: @"Category", inputColumnName: @"Category", addKeyValueAnnotationsAsText: false))
		.Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(binaryEstimator: mlContext.BinaryClassification.Trainers.FastForest(
			new FastForestBinaryTrainer.Options() { NumberOfTrees = 4, NumberOfLeaves = 4, FeatureFraction = 1F, LabelColumnName = @"Category", FeatureColumnName = @"Features" }), 
			labelColumnName: @"Category"))
		.Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: @"PredictedLabel", inputColumnName: @"PredictedLabel"));

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

void MakePredictions(IDataView? testSet = null)
{
	MLContext mlContext = new MLContext();

	// Load Trained Model
	ITransformer model = mlContext.Model.Load(Path.Combine(workspaceRelativePath, "model.zip"), out _);

	var predictionFunction = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);

	// If testSet is null, use random rows from file Luxury_Products_Apparel_Data.csv:
	if (testSet == null)
	{
		var dataView = mlContext.Data.LoadFromTextFile<ModelInput>(Path.Combine(assetsRelativePath, "Luxury_Products_Apparel_Data.csv"), hasHeader: true, separatorChar: ',');
		testSet = mlContext.Data.ShuffleRows(dataView);
		testSet = mlContext.Data.TakeRows(testSet, 3);
	}

	mlContext.Data.CreateEnumerable<ModelInput>(testSet, reuseRowObject: false).Take(3).ToList().ForEach(sample =>
	{
		var prediction = predictionFunction.Predict(sample);
		
		Console.WriteLine($"**********************************************************************");
		Console.WriteLine($"Actual Category: {sample.Category} | Predicted Category: {prediction.PredictedLabel} | Score: {prediction.Score.Max()}");
		Console.WriteLine($"**********************************************************************");
	});

}

void Evaluate(MLContext mlContext, ITransformer model, IDataView testData)
{
	var predictions = model.Transform(testData);
	var metrics = mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: "Category");
	
	Console.WriteLine();
	Console.WriteLine($"*************************************************");
	Console.WriteLine($"*       Model quality metrics evaluation         ");
	Console.WriteLine($"*------------------------------------------------");
	Console.WriteLine($"*       MacroAccuracy:      {metrics.MacroAccuracy:0.##}");
	Console.WriteLine($"*       MicroAccuracy:      {metrics.MicroAccuracy:0.##}");
	Console.WriteLine($"*       LogLoss:      {metrics.LogLoss:#.##}");
	Console.WriteLine($"*       LogLossReduction:      {metrics.LogLossReduction:#.##}");

}

public class ModelInput
{
	[LoadColumn(1)]
	[ColumnName(@"Category")]
	public string Category { get; set; }

	[LoadColumn(3)]
	[ColumnName(@"ProductName")]
	public string ProductName { get; set; }

	[LoadColumn(4)]
	[ColumnName(@"Description")]
	public string Description { get; set; }

}

public class ModelOutput
{
	[ColumnName(@"Category")]
	public uint Category { get; set; }

	[ColumnName(@"ProductName")]
	public float[] ProductName { get; set; }

	[ColumnName(@"Description")]
	public float[] Description { get; set; }

	[ColumnName(@"Features")]
	public float[] Features { get; set; }

	[ColumnName(@"PredictedLabel")]
	public string PredictedLabel { get; set; }

	[ColumnName(@"Score")]
	public float[] Score { get; set; }

}