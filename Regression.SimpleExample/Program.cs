using Microsoft.ML;
using Microsoft.ML.Data;

const string path = "..\\..\\..\\assets\\poverty.csv";

var context = new MLContext(seed: 0);

// Get the data from text file
var data = context.Data.LoadFromTextFile<Input>(path, hasHeader: true, separatorChar: ',');

// Build the pipeline
var pipeline = context.Transforms.NormalizeMinMax("PovertyRate")
	.Append(context.Transforms.Concatenate("Features", "PovertyRate"))
	.Append(context.Regression.Trainers.Ols());

// Shuffle rows
data = context.Data.ShuffleRows(data);

// Split into test and train sets
var testTrainData = context.Data.TrainTestSplit(data, testFraction: 0.2);

// Train the model on train set
var model = pipeline.Fit(testTrainData.TrainSet);

// Evaluate the model on test set
var metrics = context.Regression.Evaluate(model.Transform(testTrainData.TestSet));

Console.WriteLine();
Console.WriteLine($"*************************************************");
Console.WriteLine($"*       Model quality metrics evaluation         ");
Console.WriteLine($"*------------------------------------------------");
Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");
Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:#.##}");

// Use the model to make a prediction
var predictor = context.Model.CreatePredictionEngine<Input, Output>(model);
var input = new Input { PovertyRate = 19.7f };
var prediction = predictor.Predict(input);

Console.WriteLine($"Predicted birth rate: {prediction.BirthRate:0.##}");
Console.WriteLine($"Actual birth rate: 58.10");
Console.WriteLine();

public class Input
{
	[LoadColumn(1)]
	public float PovertyRate;

	[LoadColumn(5), ColumnName("Label")]
	public float BirthRate;
}

public class Output
{
	[ColumnName("Score")]
	public float BirthRate;
}
