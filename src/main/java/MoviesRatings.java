import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.io.IOException;
import java.util.List;

import static org.apache.spark.sql.functions.*;


public class MoviesRatings {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("Load movies")
                .appName("Downsampling")
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


        var df_mr = df_movies.join(df_ratings,df_movies.col("movieId").equalTo(df_ratings.col("movieId")));
        Dataset<Row> aggregatedDF = df_mr.groupBy("title")
                .agg(
                        min("rating").alias("min_rating"),
                        avg("rating").alias("avg_rating"),
                        max("rating").alias("max_rating"),
                        count("rating").alias("rating_cnt")
                ).orderBy(col("rating_cnt").desc());


        var avgRatings = aggregatedDF.select("avg_rating").where("rating_cnt>=0").as(Encoders.DOUBLE()).collectAsList();
        //plot_histogram(avgRatings, "Średnie wartosci ocen");
        var avgRatings2 = aggregatedDF.select("rating_cnt").where("avg_rating>=4.5").as(Encoders.DOUBLE()).collectAsList();
        //plot_histogram(avgRatings2, "Średnie wartosci ocen rating average >= 4.5");

        var avgRatings3 = aggregatedDF
                .select("rating_cnt")
                .where("rating_cnt >= 20")
                .where("avg_rating >= 3.5")
                .as(Encoders.DOUBLE()).collectAsList();
        //plot_histogram(avgRatings3, "Średnie wartosci ocen rating average >= 4.5");


        var df_release_rating = df_mr
                .withColumn("datetime", functions.from_unixtime(df_mr.col("timestamp")))
                .drop("timestamp")
                .withColumn("rating_year", regexp_extract(col("datetime"), "(\\d{4})", 1))
                .withColumn("release_to_rating_year", expr("rating_year - year"))
                .drop("rating_year");


        var avgRatings4 = df_release_rating.select("release_to_rating_year").as(Encoders.DOUBLE()).collectAsList();
        plot_histogram(avgRatings4, "Rozkład różnicy lat pomiędzy oceną a wydaniem filmu");


        df_release_rating.show();
    }

    static void plot_histogram(List<Double> x, String title) {
        Plot plt = Plot.create();
        plt.hist().add(x).bins(50);
        plt.title(title);
        try {
            plt.show();
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (PythonExecutionException e) {
            throw new RuntimeException(e);
        }
    }

}











