package Lab9;

import org.apache.spark.ml.feature.CountVectorizer;
import org.apache.spark.ml.feature.CountVectorizerModel;
import org.apache.spark.ml.feature.RegexTokenizer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import static org.apache.spark.sql.functions.size;
import static org.apache.spark.sql.functions.split;

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
        System.out.println(firstRow.get(5));




    }
}
