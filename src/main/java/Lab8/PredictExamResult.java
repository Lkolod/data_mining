package Lab8;
import java.io.IOException;
import java.util.*;

import com.github.sh0nk.matplotlib4j.NumpyUtils;
import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonConfig;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.spark.api.java.function.ForeachFunction;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.classification.BinaryLogisticRegressionTrainingSummary;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.sql.*;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.linalg.Vector;
import static org.apache.spark.sql.functions.callUDF;
import static org.apache.spark.sql.functions.col;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;


public class PredictExamResult {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("LogisticRegressionOnExam")
                .master("local")
                .getOrCreate();
        System.out.println("Using Apache Spark v" + spark.version());

        Dataset<Row> df = loadDataEgzamin(spark, "src/main/resources/Lab8/egzamin-cpp.csv");
        spark.udf().register("maxVectorElement", new MaxVectorElement(), DataTypes.DoubleType);

        Dataset<Row> df_timestamp = preprocessData(df);
        //df_timestamp.show(5);
        Dataset<Row> df_time_result = addResult(df_timestamp);
        //df_time_result.show(5);


        //printOdds(0.7787997,"OcenaC");
        //printOdds(0.0,"DateC");
        //printOdds(0.9794,"OcenaCPP");

        var model = buildRegressionModel(df_time_result);
        //df2 = preprocessData(df2);
        //var df_with_prob = calculateProbability(df2,0.7787997176612057,0.9794886589536809,0.0,-5.836640268033183);
        //df_with_prob.show(30);


        Dataset<Row> df_with_predictions = model.transform(df_time_result);

        var df_predictions = df_with_predictions
                .select("features", "rawPrediction", "probability", "prediction");
        //df_predictions.show(5);

        var df_prob = df_with_predictions
                .select("ImieNazwisko", "OcenaC", "DataC", "OcenaCpp",
                        "Egzamin", "timestamp", "Wynik", "features", "rawPrediction", "probability", "prediction");


        df_prob = df_prob.withColumn("prob", callUDF("maxVectorElement", col("probability")));
        df_prob = df_prob.drop("features", "rawPredictions", "rawPrediction", "probability");
        df_prob.show(5);

        //LogisticRegressionModel lrModel = (LogisticRegressionModel) model.stages()[1];
        //analyzePredictions(df_predictions,lrModel);

        df_prob = df_prob.repartition(1);
        df_prob.write()
                .format("csv")
                .option("header", true)
                .option("delimiter", ",")
                .mode(SaveMode.Overwrite)
                .save("src/main/resources/Lab8/egzamin-with-classification.csv");

        var model_2 = trainAndTest(df_time_result);
        BinaryLogisticRegressionTrainingSummary trainingSummary = model_2.binarySummary();
        double[] objectiveHistory = trainingSummary.objectiveHistory();
        //plotObjectiveHistory(objectiveHistory);

        Dataset<Row> roc = trainingSummary.roc();
        //roc.show();
        //plotROC(roc);

        Dataset<Row> df_fmeasures = trainingSummary.fMeasureByThreshold();
        df_fmeasures.offset(30).show();
        double maxFMeasure = df_fmeasures.select(functions.max("F-Measure")).head().getDouble(0);
        Row bestFMeasureRow = df_fmeasures.filter(col("F-Measure").equalTo(maxFMeasure)).head();
        double bestThreshold = bestFMeasureRow.getDouble(0);
        //System.out.println(bestThreshold);
        //System.out.println(maxFMeasure);
        addClassificationToGrid(spark,model_2);
    }

    public static Dataset<Row> loadDataEgzamin(SparkSession spark, String filePath) {

        StructType schema = DataTypes.createStructType(new StructField[]{
                DataTypes.createStructField(
                        "ImieNazwisko",
                        DataTypes.StringType,
                        true),
                DataTypes.createStructField(
                        "OcenaC",
                        DataTypes.DoubleType,
                        true),
                DataTypes.createStructField(
                        "DataC",
                        DataTypes.DateType,
                        true),
                DataTypes.createStructField(
                        "OcenaCpp",
                        DataTypes.DoubleType,
                        true),
                DataTypes.createStructField(
                        "Egzamin",
                        DataTypes.DoubleType,
                        true),
        });
        Dataset<Row> df = spark.read()
                .format("csv")
                .option("header", "true")
                .schema(schema)
                .option("delimiter", ";")
                .load(filePath);
        //df.show(5);
        //df.printSchema();
        return df;

    }

    public static Dataset<Row> loadDataGrid(SparkSession spark, String filePath) {

        StructType schema = DataTypes.createStructType(new StructField[]{
                DataTypes.createStructField(
                        "ImieNazwisko",
                        DataTypes.StringType,
                        true),
                DataTypes.createStructField(
                        "OcenaC",
                        DataTypes.DoubleType,
                        true),
                DataTypes.createStructField(
                        "DataC",
                        DataTypes.DateType,
                        true),
                DataTypes.createStructField(
                        "OcenaCpp",
                        DataTypes.DoubleType,
                        true),
        });
        Dataset<Row> df = spark.read()
                .format("csv")
                .option("header", "true")
                .schema(schema)
                .load(filePath);
        //df.show(5);
        //df.printSchema();
        return df;
    }

    public static Dataset<Row> preprocessData(Dataset<Row> df) {
        df = df.withColumn("timestamp", functions.unix_timestamp(df.col("DataC"), "yyyy-MM-dd").cast(DataTypes.LongType));
        return df;
    }

    public static Dataset<Row> addResult(Dataset<Row> df) {
        df = df.withColumn("Wynik", functions.expr("IF(Egzamin >= 3.0, 1, 0)"));
        return df;
    }

    public static PipelineModel buildRegressionModel(Dataset<Row> df) {

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"OcenaC", "OcenaCpp", "timestamp"})
                .setOutputCol("features");

        LogisticRegression lr = new LogisticRegression()
                .setMaxIter(1000)
                .setRegParam(0.1)
                .setElasticNetParam(0.5)
                .setFeaturesCol("features")
                .setLabelCol("Wynik");


        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{assembler, lr});
        PipelineModel model = pipeline.fit(df);
        printCoef(model);
        return model;
    }

    public static void printCoef(PipelineModel pipelineModel) {

        VectorAssembler assembler = (VectorAssembler) pipelineModel.stages()[0];
        LogisticRegressionModel lrModel = (LogisticRegressionModel) pipelineModel.stages()[1];

        String[] inputFeatures = assembler.getInputCols();
        double[] coefficients = lrModel.coefficients().toArray();
        double intercept = lrModel.intercept();

        System.out.println("logit(zdal)= ");
        for (int i = 0; i < inputFeatures.length; i++) {
            System.out.println(inputFeatures[i] + "* " + coefficients[i] + "+");

        }
        System.out.println(intercept);

    }

    public static Dataset<Row> calculateProbability(Dataset<Row> df, double OcenaC_coef, double OcenaCpp_coef, double time_coef, double intercept) {
        df = df.withColumn("logit",
                functions.expr(String.format(Locale.US, "%f * OcenaC + %f * timestamp + %f * OcenaCpp + %f",
                        OcenaC_coef, time_coef, OcenaCpp_coef, intercept)));

        df = df.withColumn("prawdopodobieństwo ",
                functions.expr("1.0 / (1.0 + EXP(-logit))"));

        return df;
    }

    public static void printOdds(double coef, String coefName) {
        // Calculate the percentage increase
        double oddsRatio = Math.exp(coef);
        double percentageIncrease = (oddsRatio - 1) * 100;
        System.out.println("Increasing the: " + coefName + " by 1 unit increases the logit by " + coef + ", " +
                "and the odds of passing by approximately " + oddsRatio + ", " +
                "which corresponds to a percentage increase of " + percentageIncrease + "%.");

    }

    private static void analyzePredictions(Dataset<Row> dfPredictions, LogisticRegressionModel lrModel) {
        dfPredictions.foreach((ForeachFunction<Row>) row -> {

            Vector features = row.getAs("features");

            double OcenaC = features.apply(0);
            double timestamp = features.apply(2);
            double OcenaCpp = features.apply(1);

            double logit = OcenaC * lrModel.coefficients().toArray()[0] +
                    timestamp * lrModel.coefficients().toArray()[2] +
                    OcenaCpp * lrModel.coefficients().toArray()[1] +
                    lrModel.intercept();

            double prob0 = 1.0 / (1.0 + Math.exp(-logit));
            double prob1 = 1.0 - prob0;

            Vector rawPrediction = row.getAs("rawPrediction");
            Vector probability = row.getAs("probability");

            int predLabel = rawPrediction.argmax();
            System.out.println("----------------------------------------");
            System.out.println("Logit: " + logit);
            System.out.println("P(0): " + prob0);
            System.out.println("P(1): " + prob1);
            System.out.println("probability " + probability);
            System.out.println("Raw Prediction: " + rawPrediction);
            System.out.println("Predicted Label: " + predLabel);
            System.out.println("----------------------------------------");
        });
    }

    static LogisticRegressionModel trainAndTest(Dataset<Row> df) {

        int splitSeed = 123;
        Dataset<Row>[] splits = df.randomSplit(new double[]{0.7, 0.3}, splitSeed);
        Dataset<Row> df_train = splits[0];
        Dataset<Row> df_test = splits[1];

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"OcenaC", "OcenaCpp", "timestamp"})
                .setOutputCol("features");

        df_train = assembler.transform(df_train);
        df_test = assembler.transform(df_test);

        LogisticRegression lr = new LogisticRegression()
                .setMaxIter(100)
                .setRegParam(0.3)
                .setFeaturesCol("features")
                .setLabelCol("Wynik");

        lr.setThreshold(0.530044012725942);

        LogisticRegressionModel lrModel = lr.fit(df_train);
        evaluateModel(lrModel,df_test);
        return lrModel;
    }
    static void evaluateModel(LogisticRegressionModel lrModel, Dataset<Row> df) {

        lrModel.setThreshold(0.530044012725942);
        Dataset<Row> predictions = lrModel.transform(df);

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("Wynik")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Accuracy = " + accuracy);

        evaluator.setMetricName("precisionByLabel");
        double precision = evaluator.evaluate(predictions);
        System.out.println("Precision = " + precision);

        evaluator.setMetricName("recallByLabel");
        double recall = evaluator.evaluate(predictions);
        System.out.println("Recall = " + recall);

        evaluator.setMetricName("f1");
        double f1 = evaluator.evaluate(predictions);
        System.out.println("F1 Score = " + f1);

        evaluator.setMetricName("weightedPrecision");
        double weightedPrecision = evaluator.evaluate(predictions);
        System.out.println("weightedPrecision = " + f1);
        evaluator.setMetricName("weightedRecall");
        double weightedRecall = evaluator.evaluate(predictions);
        System.out.println("weightedRecall = " + f1);

        BinaryClassificationEvaluator binaryEvaluator = new BinaryClassificationEvaluator()
                .setLabelCol("Wynik")
                .setRawPredictionCol("rawPrediction")
                .setMetricName("areaUnderROC");

        double Auroc = binaryEvaluator.evaluate(predictions);
        System.out.println("AUROC = " + Auroc);

        long tp = predictions.filter(col("prediction").equalTo(1.0).and(col("Wynik").equalTo(1.0))).count();
        long fp = predictions.filter(col("prediction").equalTo(1.0).and(col("Wynik").equalTo(0.0))).count();
        long tn = predictions.filter(col("prediction").equalTo(0.0).and(col("Wynik").equalTo(0.0))).count();
        long fn = predictions.filter(col("prediction").equalTo(0.0).and(col("Wynik").equalTo(1.0))).count();

        double TPR = (double) tp / (tp + fn);
        double FPR = (double) fp / (fp + tn);

        System.out.println("TPR = " + TPR);
        System.out.println("FPR = " + FPR);
    }
    public static void addClassificationToGrid(SparkSession spark, LogisticRegressionModel lrModel){
        Dataset<Row> df2 = loadDataGrid(spark, "src/main/resources/Lab8/grid.csv");
        Dataset<Row> df_preprocessed = preprocessData(df2);

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"OcenaC", "OcenaCpp", "timestamp"})
                .setOutputCol("features");

        Dataset<Row> df_transformed = assembler.transform(df_preprocessed);
        Dataset<Row> predictions = lrModel.transform(df_transformed);

        Dataset<Row> results = predictions.withColumnRenamed("prediction", "Wynik");
        results = results.drop("timestamp","features","rawPrediction","probability");
        results = results.withColumn("Wynik",
                functions.expr("IF(Wynik == 0, 'Nie zdał', 'Zdał')"));

        results.show(10);
        results = results.repartition(1);
        results.write()
                .format("csv")
                .option("header", true)
                .option("delimiter", ",")
                .mode(SaveMode.Overwrite)
                .save("src/main/resources/Lab8/grid-with-classification.csv");
    }

    public static void plotObjectiveHistory(double[] objectiveHistory) {

        Double[] doubleArray = ArrayUtils.toObject(objectiveHistory);
        List<Double> x = Arrays.asList(doubleArray);

        Plot plt = Plot.create(PythonConfig.pythonBinPathConfig("C:\\Users\\kolod\\anaconda3\\envs\\pythonProject1\\python.exe"));
        plt.plot().add(x).label("data");
        plt.title("objectiveHistory");
        try {
            plt.show();
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (PythonExecutionException e) {
            throw new RuntimeException(e);
        }
    }
    static void plotROC(Dataset<Row> roc){

        var x = roc.select("FPR").as(Encoders.DOUBLE()).collectAsList();
        var y = roc.select("TPR").as(Encoders.DOUBLE()).collectAsList();

        Plot plt = Plot.create(PythonConfig.pythonBinPathConfig("C:\\Users\\kolod\\anaconda3\\envs\\pythonProject1\\python.exe"));
        plt.plot().add(x,y).label("data");
        plt.title("ROC Curve");
        try {
            plt.show();
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (PythonExecutionException e) {
            throw new RuntimeException(e);
        }
    }


}
