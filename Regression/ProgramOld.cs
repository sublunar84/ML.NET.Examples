//using Microsoft.ML;
//using Microsoft.ML.Data;

//var projectDirectory = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../"));
//var assetsRelativePath = Path.Combine(projectDirectory, "assets");
//var workspaceRelativePath = Path.Combine(projectDirectory, "workspace");

//Console.WriteLine("Press A to train and save a model, press B to make predicitions with a saved model");

//var answer = Console.ReadKey();

//switch (answer.KeyChar.ToString().ToUpper())
//{
//    case "A":
//        TrainModel();
//        break;
//    case "B":
//        MakePredictions();
//        break;
//    default:
//        Console.WriteLine("Wrong answer!");
//        break;

//}

//void TrainModel()
//{
//    MLContext mlContext = new MLContext();

//    IDataView dataView = mlContext.Data.LoadFromTextFile<ModelInput>(Path.Combine(assetsRelativePath, /*"taxi-fare-train.csv"*/ "laptopPrice.csv"), hasHeader: true, separatorChar: ',');
//    //var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "FareAmount")
//    //    .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: "VendorId"))  
//    //    .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: "RateCode"))
//    //    .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
//    //    .Append(mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripDistance", "PaymentTypeEncoded"))
//    //    .Append(mlContext.Regression.Trainers.FastTree());

//    var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "Price")
//       .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "BrandEncoded", inputColumnName: "Brand"))
//       .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "ProcessorBrandEncoded", inputColumnName: "ProcessorBrand"))
//       .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "ProcessorNameEncoded", inputColumnName: "ProcessorName"))
//       .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RamGbEncoded", inputColumnName: "RamGb"))
//       .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "SsdEncoded", inputColumnName: "Ssd"))
//       .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "HddEncoded", inputColumnName: "Hdd"))
//       .Append(mlContext.Transforms.Concatenate("Features", "BrandEncoded", "ProcessorBrandEncoded", "ProcessorNameEncoded", "RamGbEncoded", "SsdEncoded", "HddEncoded"))
//       .Append(mlContext.Regression.Trainers.FastTree());

//    var trainedModel = pipeline.Fit(dataView);

//    SaveModel(mlContext, trainedModel, dataView);
//    Evaluate(mlContext, trainedModel);
//}

//void SaveModel(MLContext mlContext, ITransformer trainedModel, IDataView data)
//{
//    // Save Trained Model
//    mlContext.Model.Save(trainedModel, data.Schema, Path.Combine(workspaceRelativePath, "model.zip"));
//}

//void Evaluate(MLContext mlContext, ITransformer model)
//{
//    //IDataView dataView = mlContext.Data.LoadFromTextFile<ModelInput>(Path.Combine(assetsRelativePath, "taxi-fare-test.csv"), hasHeader: true, separatorChar: ',');
//    //var predictions = model.Transform(dataView);
//    //var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

//    //Console.WriteLine();
//    //Console.WriteLine($"*************************************************");
//    //Console.WriteLine($"*       Model quality metrics evaluation         ");
//    //Console.WriteLine($"*------------------------------------------------");
//    //Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");
//    //Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:#.##}");

//}

//void MakePredictions()
//{
//    MLContext mlContext = new MLContext();

//    // Load Trained Model
//    ITransformer trainedModel = mlContext.Model.Load(Path.Combine(workspaceRelativePath, "model.zip"), out var modelSchema);

//    TestSinglePrediction(mlContext, trainedModel);

//}

//void TestSinglePrediction(MLContext mlContext, ITransformer model)
//{
//    var predictionFunction = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);

//    //var sample = new ModelInput()
//    //{
//    //    VendorId = "VTS",
//    //    RateCode = "1",
//    //    PassengerCount = 1,
//    //    TripTime = 1140,
//    //    TripDistance = 3.75f,
//    //    PaymentType = "CRD",
//    //    FareAmount = 0 // To predict. Actual/Observed = 15.5
//    //};

//    //var prediction = predictionFunction.Predict(sample);

//    //Console.WriteLine($"**********************************************************************");
//    //Console.WriteLine($"Predicted fare: {prediction.FareAmount:0.####}, actual fare: 15.5");
//    //Console.WriteLine($"**********************************************************************");

    
//    var sample = new ModelInput()
//    {
//        Brand = "ASUS",
//        ProcessorBrand = "Intel",
//        ProcessorName = "Core i5",
//        RamGb = "8 GB",
//        Ssd = "256 GB",
//        Hdd = "0 GB",
//        Price = 0 // To predict. Actual/Observed = 5990 kr (46788,16 indisk rupee) https://www.netonnet.se/art/dator-surfplatta/laptop/laptop-14-16-tum/asus-vivobook-14-x415ja-eb2171w/1028720.8908/?utm_source=pricerunner.se&utm_medium=cpc&utm_campaign=prospecting_conversion_pricerunner-prisjamforelse_se&utm_content=Laptop%3ELaptop+14+-+16+tum
//    };

//    var prediction = predictionFunction.Predict(sample);

//    Console.WriteLine($"**********************************************************************");
//    Console.WriteLine($"Predicted price: {prediction.Price:0.####}, actual price: 46788,16");
//    Console.WriteLine($"**********************************************************************");
//}
//public class ModelInput
//{
//    //[LoadColumn(0)]
//    //public string? VendorId;

//    //[LoadColumn(1)]
//    //public string? RateCode;

//    //[LoadColumn(2)]
//    //public float PassengerCount;

//    //[LoadColumn(3)]
//    //public float TripTime;

//    //[LoadColumn(4)]
//    //public float TripDistance;

//    //[LoadColumn(5)]
//    //public string? PaymentType;

//    //[LoadColumn(6)]
//    //public float FareAmount;

//    [LoadColumn(0)]
//    public string? Brand;

//    [LoadColumn(1)]
//    public string? ProcessorBrand;

//    [LoadColumn(2)]
//    public string? ProcessorName;

//    [LoadColumn(4)]
//    public string? RamGb;

//    [LoadColumn(6)]
//    public string? Ssd;

//    [LoadColumn(7)]
//    public string? Hdd;

//    [LoadColumn(15)]
//    public float Price;

//    [LoadColumn(16)]
//    public string? Rating;

//    [LoadColumn(16)]
//    public float NumberOfRatings;

//    [LoadColumn(17)]
//    public float NumberOfReviews;

//    //brand,processor_brand,processor_name,processor_gnrtn,ram_gb,ram_type,ssd,hdd,os,os_bit,graphic_card_gb,weight,warranty,Touchscreen,msoffice,Price,rating,Number of Ratings,Number of Reviews
//}

//public class ModelOutput
//{
//    //    [ColumnName("Score")]
//    //    public float FareAmount;

//    [ColumnName("Score")]
//    public float Price;
//}
