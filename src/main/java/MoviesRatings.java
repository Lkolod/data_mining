import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonConfig;
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
                .appName("Movies Ratings Analysis")
                .master("local")
                .getOrCreate();
        System.out.println("Using Apache Spark v" + spark.version());

        Dataset<Row> dfMovies = loadMoviesDataset(spark, "data/movies.csv");
        Dataset<Row> dfRatings = loadRatingsDataset(spark, "data/ratings.csv");
        Dataset<Row> dfMR = preprocessMovies(dfMovies);
        Dataset<Row> aggregatedDF = aggregateRatings(dfMR, dfRatings);

        List<Double> avgRatings = getAverageRatings(aggregatedDF);
        //plotHistogram(avgRatings, "Average Ratings Distribution");

        List<Double> avgRating2 = getTreshold1(aggregatedDF, 4.5);
        //plotHistogram(avgRating2, "Average Ratings Distribution (Rating Average >= 4.5)");

        List<Double> avgRaring3 = getTreshold2(aggregatedDF, 3.5, 20);
        //plotHistogram(avgRaring3, "Average Ratings Distribution (Rating Average >= 3.5 with Minimum Ratings 20)");

        Dataset<Row> dfReleaseRating = preprocessReleaseRating(dfMR, dfRatings);
        Dataset<Row> downsamplingDF = downsampleData(dfReleaseRating, 0.005);
        List<Double> releaseRatingYearsDifference = getReleaseRatingYearsDifference(downsamplingDF);
        //plotHistogram(releaseRatingYearsDifference, "Distribution of Difference in Years between Rating and Release Year");

        Dataset<Row> dfRatingGrouped = groupByReleaseRatingYear(dfReleaseRating);
        Dataset<Row> dfMR2 = filterNullReleaseRatingYears(dfRatingGrouped);

        List<Double> release_to_rating_values = dfMR2.select("release_to_rating_year").as(Encoders.DOUBLE())
                .collectAsList();

        List<Double> realease_rating_count = dfMR2.select("count").as(Encoders.DOUBLE())
                .collectAsList();

        dfMR2.show();
        plot_histogram2(release_to_rating_values,realease_rating_count,"rozklad roznicy lat pomiedzy ocena a wydaniem filmu");
    }

    static Dataset<Row> loadMoviesDataset(SparkSession spark, String path) {
        StructType schema = DataTypes.createStructType(new StructField[]{
                DataTypes.createStructField("MovieId", DataTypes.IntegerType, true),
                DataTypes.createStructField("title", DataTypes.StringType, false),
                DataTypes.createStructField("genres", DataTypes.StringType, false),
        });

        return spark.read()
                .format("csv")
                .option("header", "true")
                .schema(schema)
                .load(path);
    }

    static Dataset<Row> loadRatingsDataset(SparkSession spark, String path) {
        StructType schema = DataTypes.createStructType(new StructField[] {
                DataTypes.createStructField("userId", DataTypes.IntegerType, true),
                DataTypes.createStructField("MovieId", DataTypes.IntegerType, false),
                DataTypes.createStructField("rating", DataTypes.DoubleType, false),
                DataTypes.createStructField("timestamp", DataTypes.IntegerType, false)
        });

        return spark.read()
                .format("csv")
                .option("header", "true")
                .schema(schema)
                .load(path);
    }

    static Dataset<Row> preprocessMovies(Dataset<Row> dfMovies) {
        return dfMovies
                .withColumn("title2",
                        when(regexp_extract(col("title"),"^(.*?)\\s*\\((\\d{4})\\)\\s*$",1).equalTo("")
                                ,col("title"))
                                .otherwise(regexp_extract(col("title"),"^(.*?)\\s*\\((\\d{4})\\)\\s*$",1)))
                .withColumn("year", regexp_extract(dfMovies.col("title"), "^(.+)\\((\\d{4})\\)$", 2))
                .drop("title")
                .withColumnRenamed("title2", "title");
    }

    static Dataset<Row> aggregateRatings(Dataset<Row> dfMovies, Dataset<Row> dfRatings) {
        Dataset<Row> dfMR = dfMovies.join(dfRatings, dfMovies.col("movieId").equalTo(dfRatings.col("movieId")));
        return dfMR.groupBy("title")
                .agg(
                        min("rating").alias("min_rating"),
                        avg("rating").alias("avg_rating"),
                        max("rating").alias("max_rating"),
                        count("rating").alias("rating_cnt")
                ).orderBy(col("rating_cnt").desc());
    }

    static List<Double> getAverageRatings(Dataset<Row> aggregatedDF) {
        return aggregatedDF.select("avg_rating").where("rating_cnt>=0").as(Encoders.DOUBLE()).collectAsList();
    }

    static List<Double> getTreshold1(Dataset<Row> aggregatedDF, double threshold) {
        return aggregatedDF.select("rating_cnt").where("avg_rating>=" + threshold).as(Encoders.DOUBLE()).collectAsList();
    }

    static List<Double> getTreshold2(Dataset<Row> aggregatedDF, double threshold, int minRatings) {
        return aggregatedDF.select("rating_cnt").where("rating_cnt >= " + minRatings + " AND avg_rating >= " + threshold).as(Encoders.DOUBLE()).collectAsList();
    }

    static Dataset<Row> preprocessReleaseRating(Dataset<Row> dfMovies, Dataset<Row> dfRatings) {
        return dfMovies.join(dfRatings, dfMovies.col("movieId").equalTo(dfRatings.col("movieId")))
                .withColumn("datetime", functions.from_unixtime(dfRatings.col("timestamp")))
                .drop("timestamp")
                .withColumn("rating_year", regexp_extract(col("datetime"), "(\\d{4})", 1))
                .withColumn("release_to_rating_year", expr("rating_year - year"))
                .drop("rating_year");
    }

    static Dataset<Row> downsampleData(Dataset<Row> dfReleaseRating, double fraction) {
        return dfReleaseRating.sample(fraction);
    }

    static List<Double> getReleaseRatingYearsDifference(Dataset<Row> downsamplingDF) {
        return downsamplingDF.select("release_to_rating_year").as(Encoders.DOUBLE()).collectAsList();
    }

    static void plotHistogram(List<Double> x, String title) {
        Plot plt = Plot.create(PythonConfig.pythonBinPathConfig("C:\\Users\\kolod\\anaconda3\\envs\\pythonProject1\\python.exe"));
        plt.hist().add(x).bins(50);
        plt.title(title);
        try {
            plt.show();
        } catch (IOException | PythonExecutionException e) {
            throw new RuntimeException(e);
        }
    }

    static Dataset<Row> groupByReleaseRatingYear(Dataset<Row> dfReleaseRating) {
        Dataset<Row> dfRatingGrouped = dfReleaseRating.groupBy("release_to_rating_year")
                .agg(count("release_to_rating_year").alias("count"));
        return dfRatingGrouped.orderBy(col("release_to_rating_year").asc());
    }

    static Dataset<Row> filterNullReleaseRatingYears(Dataset<Row> dfRatingGrouped) {
        return dfRatingGrouped.filter("release_to_rating_year != -1 AND release_to_rating_year IS NOT NULL");
    }
    static void plot_histogram2(List<Double> x, List<Double> weights, String title) {
        Plot plt = Plot.create(PythonConfig.pythonBinPathConfig("C:\\Users\\kolod\\anaconda3\\envs\\pythonProject1\\python.exe"));
        plt.hist().add(x).weights(weights).bins(50);
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
