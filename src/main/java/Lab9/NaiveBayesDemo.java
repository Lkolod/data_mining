package Lab9;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.Matrix;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import java.util.*;



public class NaiveBayesDemo {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("Recognition Decision Tree")
                .master("local")
                .getOrCreate();

        TestNaiveBayes(spark);
        String filename = "two-books-all-1000-10-stem.csv";
        //var metrics = performGridSearchCV(spark,filename);

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

        for (String filename2 : filenames) {
            Dataset<Row> metricsDF = performGridSearchCV(spark, filename2);
            if (mergedMetricsDF == null) {
                mergedMetricsDF = metricsDF;
            } else {
                mergedMetricsDF = mergedMetricsDF.unionAll(metricsDF);
            }
        }
        mergedMetricsDF.show();
    }
    public static void TestNaiveBayes(SparkSession spark) {
        StructType schema = new StructType()
                .add("author", DataTypes.StringType, false)
                .add("content", DataTypes.StringType, false);

        List<Row> rows = Arrays.asList(
                RowFactory.create("Ala", "aaa aaa bbb ccc"),
                RowFactory.create("Ala", "aaa bbb ddd"),
                RowFactory.create("Ala", "aaa bbb"),
                RowFactory.create("Ala", "aaa bbb bbb"),
                RowFactory.create("Ola", "aaa ccc ddd"),
                RowFactory.create("Ola", "bbb ccc ddd"),
                RowFactory.create("Ola", "ccc ddd eee")
        );

        var df = spark.createDataFrame(rows, schema);
        String sep = "[\\s\\p{Punct}—…”„]+";
        RegexTokenizer tokenizer = new RegexTokenizer()
                .setInputCol("content")
                .setOutputCol("words")
                .setPattern(sep);
        df = tokenizer.transform(df);
        df.show();

        System.out.println("-----------");
        // Convert to BoW with CountVectorizer
        CountVectorizer countVectorizer = new CountVectorizer()
                .setInputCol("words")
                .setOutputCol("features")
                .setVocabSize(10_000)  // Set the maximum size of the vocabulary
                .setMinDF(1)     // Set the minimum number of documents in which a term must appear
                ;

        // Fit the model and transform the DataFrame
        CountVectorizerModel countVectorizerModel = countVectorizer.fit(df);
        df = countVectorizerModel.transform(df);


        // Prepare the data: index the label column
        StringIndexer labelIndexer = new StringIndexer()
                .setInputCol("author")
                .setOutputCol("label");

        StringIndexerModel labelModel = labelIndexer.fit(df);
        df = labelModel.transform(df);
        df.show();

        NaiveBayes nb = new NaiveBayes()
                .setLabelCol("label")
                .setFeaturesCol("features")
                .setModelType("multinomial")
                .setSmoothing(0.01);
        System.out.println(nb.explainParams());

        NaiveBayesModel model = nb.fit(df);

        String[] vocab = countVectorizerModel.vocabulary();
        String[] labels = labelModel.labels();

        /*System.out.println("Vocabulary:");
        for (String word : vocab) {
            System.out.println(word);
        }

        System.out.println("Labels:");
        for (String label : labels) {
            System.out.println(label);
        }*/

        // Get the theta matrix from the model
        Matrix theta = model.theta();

        // Print conditional probabilities (likelihoods) based on the theta matrix
        for (int i = 0; i < theta.numRows(); i++) {
            for (int j = 0; j < theta.numCols(); j++) {
                double logLikelihood = theta.apply(i, j);
                double likelihood = Math.exp(logLikelihood);
                System.out.printf("P(%s|%s)=%.6f (log=%.6f)%n", vocab[j], labels[i], likelihood, logLikelihood);
            }
        }
        double[] priors = model.pi().toArray();
        System.out.println("Prior Probabilities:");
        for (int i = 0; i < priors.length; i++) {
            double prior = Math.exp(priors[i]);
            System.out.printf("P(%s)=%.6f (log=%.6f)%n", labels[i], prior, priors[i]);
        }


        //Prediction
        var testData = new DenseVector(new double[]{1, 0, 2, 1, 1});
        var proba = model.predictRaw(testData);
        System.out.println("Pr:[" + Math.exp(proba.apply(0)) + ", " + Math.exp(proba.apply(1)));
        var predLabel = model.predict(testData);
        System.out.println(predLabel);

        var p0 = proba.apply(0);
        var p1 = proba.apply(1);

        System.out.printf(Locale.US,"log(p0)=%g p0=%g log(p1)=%g p1=%g\n",
                p0,Math.exp(p0),
                p1,Math.exp(p1));
        System.out.println("Wynik klasyfikacj:"+(p0>p1?0:1));

    }


    private static Dataset<Row> performGridSearchCV(SparkSession spark, String filename){
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

        NaiveBayes nb = new NaiveBayes()
                .setLabelCol("label")
                .setFeaturesCol("features")
                .setSmoothing(0.2);

        CountVectorizer countVectorizer = new CountVectorizer()
                .setInputCol("words")
                .setOutputCol("features")
                .setVocabSize(10_000)  // Set the maximum size of the vocabulary
                .setMinDF(2);           // Set the minimum number of documents in which a term must appear

        var scalaIterable = scala.jdk.CollectionConverters.
                IterableHasAsScala(Arrays.asList("multinomial", "gaussian")).asScala();

        ParamMap[] paramGrid = new ParamGridBuilder().build();

        StringIndexer indexer = new StringIndexer()
                .setInputCol("author")
                .setOutputCol("label");

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[] {tokenizer, countVectorizer,indexer, nb});

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("f1");

        CrossValidator cv = new CrossValidator()
                .setEstimator(pipeline)
                .setEvaluator(evaluator)
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(3)
                .setParallelism(8);

        CrossValidatorModel cvModel = cv.fit(df_train);
        double[] avgMetrics = cvModel.avgMetrics();
        double average = Arrays.stream(avgMetrics).average().orElse(Double.NaN);
        System.out.println(average);


        PipelineModel bestModel = (PipelineModel) cvModel.bestModel();
        for(var s:bestModel.stages()){
            System.out.println(s);
        }

        Dataset<Row> predictions = bestModel.transform(df_test);
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
