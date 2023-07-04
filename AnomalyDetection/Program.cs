using Microsoft.ML;
using Microsoft.ML.Data;
using PLplot;

var projectDirectory = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../"));
var assetsRelativePath = Path.Combine(projectDirectory, "assets");

// Set the environment variable for PLplot
Environment.SetEnvironmentVariable("PLPLOT_LIB", Path.Combine(AppContext.BaseDirectory, "runtimes", "win-x64", "native"));


// create the machine learning context
var context = new MLContext();

// load the data file
Console.WriteLine("Loading data...");
var dataView = context.Data.LoadFromTextFile<WeatherData>(path: Path.Combine(assetsRelativePath, "Canadian_climate_history.csv"), hasHeader: true, separatorChar: ',');

// get an array of data points
var values = context.Data.CreateEnumerable<WeatherData>(dataView, reuseRowObject: false).ToArray();

// Plot data
//PlotData(values);
// plot the data

var pl = new PLStream();
pl.sdev("pngcairo");                // png rendering
pl.sfnam(Path.Combine(assetsRelativePath, "data.png"));                 // output filename
pl.spal0("cmap0_alternate.pal");   // alternate color palette
pl.init();
pl.env(
	28000, 28200,                          // x-axis range. Total about 30 000 days.
	-50, 50,                        // y-axis range
	AxesScale.Independent,               // scale x and y independently
	AxisBox.BoxTicksLabelsAxes);             // draw box, ticks, and num ticks
pl.lab(
	"Day",                                 // x-axis label
	"Mean temperature Montreal",              // y-axis label
	"Mean temperature Montreal over time");   // plot title
pl.line(
	(from x in Enumerable.Range(0, values.Count()) select (double)x).ToArray(), 
	(from p in values select (double)p.MeanTemperatureMontreal).ToArray()
);

// build a training pipeline for detecting spikes
var pipeline = context.Transforms.DetectSpikeBySsa(
	nameof(TemperaturePrediction.Prediction),
	nameof(WeatherData.MeanTemperatureMontreal),
	confidence: 99.0,
	pvalueHistoryLength: 730, // 30
	trainingWindowSize: 29221, // 90
	seasonalityWindowSize: 365); // 30

// train the model
Console.WriteLine("Detecting spikes...");
var model = pipeline.Fit(dataView);

// predict spikes in the data
var transformed = model.Transform(dataView);
var predictions = context.Data.CreateEnumerable<TemperaturePrediction>(transformed, reuseRowObject: false).ToArray();

// find the spikes in the data
var spikes = (from i in Enumerable.Range(0, predictions.Count())
	where predictions[i].Prediction[0] == 1
	select (Day: i, Temperature: values[i].MeanTemperatureMontreal));

var spikesWithDates = (from s in spikes select (Date: values[s.Day].Time, Temperature: s.Temperature)).ToList();

// plot the spikes
pl.col0(2);     // blue color
pl.schr(3, 3);  // scale characters
pl.string2(
	(from s in spikes select (double)s.Day).ToArray(),
	(from s in spikes select (double)s.Temperature + 30).ToArray(),
	"|");

pl.eop();

void PlotData(WeatherData[] values)
{
	// plot the data
	//var pl = new PLStream();
	//pl.sdev("pngcairo");                // png rendering
	//pl.sfnam(Path.Combine(assetsRelativePath, "data.png"));                 // output filename
	//pl.spal0("cmap0_alternate.pal");   // alternate color palette
	//pl.init();
	//pl.env(
	//	29000, 29365,                          // x-axis range. Total about 30 000 days.
	//	-50,50 ,                        // y-axis range
	//	AxesScale.Independent,               // scale x and y independently
	//	AxisBox.BoxTicksLabelsAxes);             // draw box, ticks, and num ticks
	//pl.lab(
	//	"Day",                                 // x-axis label
	//	"Mean temperature Calgary",              // y-axis label
	//	"Mean temperature Calgary over time");   // plot title
	//pl.line(
	//	(from x in Enumerable.Range(0, values.Count()) select (double)x).ToArray(),
	//	(from p in values select (double)p.MeanTemperatureCalgary).ToArray()
	//);
	//pl.eop();

}

/// <summary>
/// The WeatherData class contains one weather record.
/// </summary>
public class WeatherData
{
	[LoadColumn(0)] public DateTime Time { get; set; }
	[LoadColumn(9)] public float MeanTemperatureMontreal { get; set; }
	[LoadColumn(10)] public float TotalPrecipitationMontreal { get; set; }
}

/// <summary>
/// The TemperaturePrediction class contains one temperature spike prediction.
/// </summary>
public class TemperaturePrediction
{
	[VectorType(3)]
	public double[] Prediction { get; set; }
}



