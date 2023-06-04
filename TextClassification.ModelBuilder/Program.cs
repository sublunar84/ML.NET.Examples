// See https://aka.ms/new-console-template for more information
using TextClassification_ModelBuilder;

//Load sample data
var sampleData = new TextClassification.ModelInput()
{
    Column1 = 13670F,
    SubCategory = @"Socks",
    ProductName = @"""Falke - Lhasa Wool And Cashmere-blend Socks - Mens - Navy""",
    Description = @"""Falke - Casual yet luxurious, Falke's dark navy Lhasa socks are woven from a mid-weight wool and cashmere-blend that's naturally insulating. They have a soft rib for comfort and reinforced stress zones for durability. Wear them to round off endless looks.""",
};

//Load model and predict output
var result = TextClassification.Predict(sampleData);
Console.WriteLine($"Predicted Label value {result.PredictedLabel}, Score: {result.Score.Max()}");