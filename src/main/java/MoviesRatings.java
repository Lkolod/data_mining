import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import static org.apache.spark.sql.functions.*;


public class MoviesRatings {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("Load movies")
                .master("local")
                .getOrCreate();
        System.out.println("Using Apache Spark v" + spark.version());

        StructType schema = DataTypes.createStructType(new StructField[]{
                DataTypes.createStructField(
                        "MovieId",
                        DataTypes.IntegerType,
                        true),
                DataTypes.createStructField(
                        "title",
                        DataTypes.StringType,
                        false),
                DataTypes.createStructField(
                        "genres",
                        DataTypes.StringType,
                        false),

        });

        Dataset<Row> df = spark.read()
                .format("csv")
                .option("header", "true")
                .schema(schema)
                .load("data/movies.csv");

        var df_movies = df
                .withColumn("title2", regexp_extract(df.col("title"), "^(.+)\\((\\d{4})\\)$", 1))
                .withColumn("year", regexp_extract(df.col("title"), "^(.+)\\((\\d{4})\\)$", 2))
                .drop("title")
                .withColumnRenamed("title2", "title");


        StructType schema2 = DataTypes.createStructType(new StructField[] {
                DataTypes.createStructField(
                        "userId",
                        DataTypes.IntegerType,
                        true),
                DataTypes.createStructField(
                        "MovieId",
                        DataTypes.IntegerType,
                        false),
                DataTypes.createStructField(
                        "rating",
                        DataTypes.DoubleType,
                        false),
                DataTypes.createStructField(
                        "timestamp",
                        DataTypes.IntegerType,
                        false),        });

        Dataset<Row> df_ratings = spark.read()
                .format("csv")
                .option("header", "true")
                .schema(schema2)
                .load("data/ratings.csv");


        var df_mr = df_movies.join(df_ratings,"movieId","inner");
        Dataset<Row> aggregatedDF = df_mr.groupBy("title")
                .agg(
                        min("rating").alias("min_rating"),
                        avg("rating").alias("avg_rating"),
                        max("rating").alias("max_rating"),
                        count("rating").alias("rating_cnt")
                ).orderBy(col("rating_cnt").desc());
        aggregatedDF.show();



        //Dataset<Row> grouped_mr = df_mr.groupBy("title",).agg(count("columnNameToAggregate"));
    }

}











