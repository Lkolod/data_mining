package Lab9;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;

import java.util.*;

import static org.apache.spark.sql.functions.size;
import static org.apache.spark.sql.functions.split;

public class AuthorRecognitionCVDecisionTree {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("Recognition Decision Tree")
                .master("local")
                .getOrCreate();

        String filenames[]={
                "two-books-all-1000-1-stem.csv",
                "two-books-all-1000-3-stem.csv",
                "two-books-all-1000-5-stem.csv",
                "two-books-all-1000-10-stem.csv",
                "five-books-all-1000-1-stem.csv",
                "five-books-all-1000-3-stem.csv",
                "five-books-all-1000-5-stem.csv",
                "five-books-all-1000-10-stem.csv",
        };
        Dataset<Row> mergedMetricsDF = null;

        for (String filename : filenames) {
            Dataset<Row> metricsDF = performCV(spark, filename);
            if (mergedMetricsDF == null) {
                mergedMetricsDF = metricsDF;
            } else {
                mergedMetricsDF = mergedMetricsDF.unionAll(metricsDF);
            }
        }
        mergedMetricsDF.show();
    }
    private static Dataset<Row> performCV(SparkSession spark, String filename){
        //author,work,content,content_stemmed
        Dataset<Row> df = spark.read().format("csv")
                .option("header", "true")
                .option("delimiter", ",")
                .option("quote", "\'")
                .option("inferschema", "true")
                .load("src/main/resources/lab9/" +filename);

        var splits = df.randomSplit(new double[]{0.8,0.2});
        var df_train = splits[0];
        var df_test = splits[1];

        String sep = "[\\s\\p{Punct}—…”„]+";
        RegexTokenizer tokenizer = new RegexTokenizer()
                .setInputCol("content")
                .setOutputCol("words")
                .setPattern(sep);

        CountVectorizer countVectorizer = new CountVectorizer()
                .setInputCol("words")
                .setOutputCol("features")
                .setVocabSize(10_000)  // Set the maximum size of the vocabulary
                .setMinDF(2);           // Set the minimum number of documents in which a term must appear

        StringIndexer indexer = new StringIndexer()
                .setInputCol("author")
                .setOutputCol("label");

        DecisionTreeClassifier dt = new DecisionTreeClassifier()
                .setLabelCol("label")
                .setFeaturesCol("features")
                .setImpurity("gini")
                .setMaxDepth(30);

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[] {tokenizer, countVectorizer,indexer, dt});

        var model = pipeline.fit(df_train);
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("f1");


        Dataset<Row> predictions = model.transform(df_test);
        double f1 = evaluator.evaluate(predictions);
        evaluator.setMetricName("weightedPrecision");
        double weightedPrecision = evaluator.evaluate(predictions);
        evaluator.setMetricName("weightedRecall");
        double weightedRecall = evaluator.evaluate(predictions);
        evaluator.setMetricName("accuracy");
        double accuracy = evaluator.evaluate(predictions);

        System.out.println("Accuracy: " + accuracy);
        System.out.println("Weighted Precision: " + weightedPrecision);
        System.out.println("Weighted Recall: " + weightedRecall);
        System.out.println("F1 Score: " + f1);

        Dataset<Row> metricsDF = spark.createDataFrame(
                List.of(
                        RowFactory.create(filename, accuracy, weightedPrecision, weightedRecall, f1)
                ),
                DataTypes.createStructType(List.of(
                        DataTypes.createStructField("Filename", DataTypes.StringType, false),
                        DataTypes.createStructField("Accuracy", DataTypes.DoubleType, false),
                        DataTypes.createStructField("Weighted Precision", DataTypes.DoubleType, false),
                        DataTypes.createStructField("Weighted Recall", DataTypes.DoubleType, false),
                        DataTypes.createStructField("F1 Score", DataTypes.DoubleType, false)
                ))
        );

        return metricsDF;
    }
}
