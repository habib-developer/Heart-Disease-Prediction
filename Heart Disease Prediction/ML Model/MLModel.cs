using Heart_Disease_Prediction.ML_Model.DataStructures;
using Microsoft.ML;
using System.IO;
using System.Linq;

namespace Heart_Disease_Prediction.ML_Model
{
    public class MLModel
    {
        private readonly MLContext mlContext;
        private ITransformer trainedModel = null;

        private static string BaseDatasetsRelativePath = @"../../../ML Model/Data";
        private static string TrainDataRelativePath = $"{BaseDatasetsRelativePath}/HeartTraining.csv";

        private static string TrainDataPath = GetAbsolutePath(TrainDataRelativePath);

        public MLModel()
        {
            mlContext = new MLContext();
        }
        public void Build()
        {
            // STEP 1: Common data loading configuration
            var trainingDataView = mlContext.Data.LoadFromTextFile<HeartData>(TrainDataPath, hasHeader: true, separatorChar: ';');

            // STEP 2: Concatenate the features and set the training algorithm
            var pipeline = mlContext.Transforms.Concatenate("Features", "Age", "Sex", "Cp", "TrestBps", "Chol", "Fbs", "RestEcg", "Thalac", "Exang", "OldPeak", "Slope", "Ca", "Thal")
                            .Append(mlContext.BinaryClassification.Trainers.FastTree(labelColumnName: "Label", featureColumnName: "Features"));
            trainedModel = pipeline.Fit(trainingDataView);
        }
        public HeartPrediction Consume(HeartData input)
        {
            var predictionEngine = mlContext.Model.CreatePredictionEngine<HeartData, HeartPrediction>(trainedModel);
            return predictionEngine.Predict(input);
        }
        private static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;

        }
    }
}
