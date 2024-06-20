package Lab9;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.linalg.SparseVector;
import org.apache.spark.ml.feature.CountVectorizer;
import org.apache.spark.ml.feature.CountVectorizerModel;
import org.apache.spark.ml.feature.RegexTokenizer;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.StringIndexer;

import java.util.*;

import static org.apache.spark.sql.functions.*;

public class AuthorRecognitionDecisionTree {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("Recognition Decision Tree")
                .master("local")
                .getOrCreate();

        Dataset<Row> df = spark.read().format("csv")
                .option("header", "true")
                .option("delimiter", ",")
                .option("quote", "\'")
                .option("inferschema", "true")
                .load("src/main/resources/lab9/two-books-all-1000-10-stem.csv");

        //df.show();
        var author_work = df.select("author", "work").distinct();
        //author_work.show();
        var df_work_distribution = df.groupBy("author", "work").count();
        //df_work_distribution.show();
        var df_with_len = df.withColumn("wordCount", size(split(df.col("content"), " ")));
        //df_with_len.show();

        var avg_words = df_with_len.groupBy("author", "work").avg("wordCount");
        //avg_words.show();

        String sep = "[\\s\\p{Punct}—…”„]+";
        RegexTokenizer tokenizer = new RegexTokenizer()
                .setInputCol("content")
                .setOutputCol("words")
                .setPattern(sep);
        var df_tokenized = tokenizer.transform(df);
        //df_tokenized.show();


        CountVectorizer countVectorizer = new CountVectorizer()
                .setInputCol("words")
                .setOutputCol("features")
                .setVocabSize(10_000)  // Set the maximum size of the vocabulary
                .setMinDF(2);           // Set the minimum number of documents in which a term must appear

        CountVectorizerModel countVectorizerModel = countVectorizer.fit(df_tokenized);

        Dataset<Row> df_bow = countVectorizerModel.transform(df_tokenized);
        //df_bow.select("words", "features").show(5);

        Row firstRow = df_bow.first();
        //System.out.println("Words: " + firstRow.getList(firstRow.fieldIndex("words")));
        //System.out.println("Features: " +  firstRow.get(firstRow.fieldIndex("features")));

        SparseVector features = (SparseVector) firstRow.get(firstRow.fieldIndex("features"));
        int[] indices = features.indices();

        var vocabulary = countVectorizerModel.vocabulary();
        for (int index : indices) {
            String word = vocabulary[index];
            double count = features.apply(index);
            //System.out.println(word + " -> " + count);
        }

        StringIndexer indexer = new StringIndexer()
                .setInputCol("author")
                .setOutputCol("label");

        StringIndexerModel labelModel = indexer.fit(df_bow);
        df_bow = labelModel.transform(df_bow);
        //df_bow.show(20);

        DecisionTreeClassifier dt = new DecisionTreeClassifier()
                .setLabelCol("label")
                .setFeaturesCol("features")
                .setImpurity("gini")  // lub entropy
                .setMaxDepth(30);

        DecisionTreeClassificationModel model = dt.fit(df_bow);

        Dataset<Row> df_predictions = model.transform(df_bow);
        //df_predictions.show();

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("f1");

        double f1 = evaluator.evaluate(df_predictions);
        //System.out.println("F1 score: " + f1);
        evaluator.setMetricName("accuracy");
        double accuracy = evaluator.evaluate(df_predictions);
        //System.out.println("accuracy: " + accuracy);


        SparseVector fi = (SparseVector) model.featureImportances();
        //System.out.println(fi);
        Map<String, Double> wordImportanceMap = new HashMap<>();

        for (int index : fi.indices()) {
            var word = vocabulary[index];
            var importance = fi.apply(index);
            wordImportanceMap.put(word, importance);
        }
        List<Map.Entry<String, Double>> list = new ArrayList<>(wordImportanceMap.entrySet());
        list.sort(Map.Entry.comparingByValue(Collections.reverseOrder()));

        for (Map.Entry<String,Double> entry: list){
            System.out.println(entry.getKey() + " --> " + entry.getValue());
        }
        PipelineModel model_grid = performGridSearchCV(spark,"five-books-all-1000-10-stem.csv");

    }
    private static PipelineModel performGridSearchCV(SparkSession spark, String filename){
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

        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(countVectorizer.vocabSize(), new int[] {100, 1000,10_000})
                .addGrid(dt.maxDepth(), new int[] {10, 20,30})
                .build();

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

        return bestModel;

    }
}
