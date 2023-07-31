using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Vision;

var projectDirectory = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../"));
var assetsRelativePath = Path.Combine(projectDirectory, "assets");
var workspaceRelativePath = Path.Combine(projectDirectory, "workspace");
var testRelativePath = Path.Combine(projectDirectory, "test");

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

    IEnumerable<ImageData> images = LoadImagesFromDirectory(folder: assetsRelativePath, useFolderNameAsLabel: true);
    IDataView imageData = mlContext.Data.LoadFromEnumerable(images);

    IDataView shuffledData = mlContext.Data.ShuffleRows(imageData);

    IDataView preProcessedData = GetPreProcessedData(mlContext, shuffledData);
    
    TrainTestData trainSplit = mlContext.Data.TrainTestSplit(data: preProcessedData, testFraction: 0.3);
    TrainTestData validationTestSplit = mlContext.Data.TrainTestSplit(trainSplit.TestSet);

    IDataView trainSet = trainSplit.TrainSet;
    IDataView validationSet = validationTestSplit.TrainSet;
    IDataView testSet = validationTestSplit.TestSet;

    var classifierOptions = new ImageClassificationTrainer.Options()
    {
        FeatureColumnName = "Image",
        LabelColumnName = "LabelAsKey",
        ValidationSet = validationSet,
        Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
        MetricsCallback = (metrics) => Console.WriteLine(metrics),
        TestOnTrainSet = false,
        ReuseTrainSetBottleneckCachedValues = true,
        ReuseValidationSetBottleneckCachedValues = true
    };

    var trainingPipeline = mlContext.MulticlassClassification.Trainers.ImageClassification(classifierOptions)
        .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

    ITransformer trainedModel = trainingPipeline.Fit(trainSet);

    SaveModel(mlContext, trainedModel, trainSet);
    
    IDataView predictions = trainedModel.Transform(testSet);
    MulticlassClassificationMetrics metrics =
	    mlContext.MulticlassClassification.Evaluate(predictions,
		    labelColumnName: "LabelAsKey",
		    predictedLabelColumnName: "PredictedLabel");

    Console.WriteLine();
    Console.WriteLine($"*************************************************");
    Console.WriteLine($"*       Model quality metrics evaluation         ");
    Console.WriteLine($"*------------------------------------------------");
	Console.WriteLine($"LogLoss is: {metrics.LogLoss}");
    Console.WriteLine($"PerClassLogLoss is: {string.Join(" , ", metrics.PerClassLogLoss.Select(c => c.ToString()))}");

}

IDataView GetPreProcessedData(MLContext mlContext, IDataView data)
{
	var preprocessingPipeline = mlContext.Transforms.Conversion.MapValueToKey(
			inputColumnName: "Label",
			outputColumnName: "LabelAsKey")
		.Append(mlContext.Transforms.LoadRawImageBytes(
			outputColumnName: "Image",
			imageFolder: assetsRelativePath,
			inputColumnName: "ImagePath"));

	return preprocessingPipeline
		.Fit(data)
		.Transform(data);
}

void SaveModel(MLContext mlContext, ITransformer trainedModel, IDataView data)
{
    // Save Trained Model
    mlContext.Model.Save(trainedModel, data.Schema, Path.Combine(workspaceRelativePath, "model.zip"));
}

static void OutputPrediction(ModelOutput prediction)
{
    string imageName = Path.GetFileName(prediction.ImagePath);
    Console.WriteLine($"Image: {imageName} | Actual Value: {prediction.Label} | Predicted Value: {prediction.PredictedLabel} | Score: {prediction.Score.Max()}");
}

IEnumerable<ImageData> LoadImagesFromDirectory(string folder, bool useFolderNameAsLabel = true)
{
    var files = Directory.GetFiles(folder, "*",
    searchOption: SearchOption.AllDirectories);

    foreach (var file in files)
    {
        if ((Path.GetExtension(file) != ".jpg") && (Path.GetExtension(file) != ".png"))
            continue;

        var label = Path.GetFileName(file);

        if (useFolderNameAsLabel)
            label = Directory.GetParent(file).Name;
        else
        {
            for (int index = 0; index < label.Length; index++)
            {
                if (!char.IsLetter(label[index]))
                {
                    label = label.Substring(0, index);
                    break;
                }
            }
        }

        yield return new ImageData()
        {
            ImagePath = file,
            Label = label
        };
    }
}

void MakePredictions()
{
    MLContext mlContext = new MLContext();

    // Load Trained Model
    ITransformer trainedModel = mlContext.Model.Load(Path.Combine(workspaceRelativePath, "model.zip"), out _);

	// Load image(s) from disk
	IEnumerable<ImageData> images = LoadImagesFromDirectory(testRelativePath, useFolderNameAsLabel: true);
	IDataView imageData = mlContext.Data.LoadFromEnumerable(images);

	IDataView preProcessedData = GetPreProcessedData(mlContext, imageData);

	IEnumerable<ModelInput> imageInputs = mlContext.Data.CreateEnumerable<ModelInput>(preProcessedData, reuseRowObject: true);

	var predictor = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(trainedModel);

	Console.WriteLine("Classifying multiple images");
	foreach (var image in imageInputs)
    {
	    OutputPrediction(predictor.Predict(image));
    }
}

public class ImageData
{
    public string ImagePath { get; set; }

    public string Label { get; set; }
}

class ModelInput
{
	public byte[] Image { get; set; }

	public UInt32 LabelAsKey { get; set; }

	public string ImagePath { get; set; }

	public string Label { get; set; }
}

public class ModelOutput
{
    public string ImagePath { get; set; }

    public string Label { get; set; }

    public string PredictedLabel { get; set; }

    public float[]? Score { get; set; }
}