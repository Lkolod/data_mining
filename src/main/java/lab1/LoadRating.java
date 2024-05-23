import com.github.sh0nk.matplotlib4j.NumpyUtils;
import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.io.IOException;

import static org.apache.spark.sql.functions.*;

public class LoadRating {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("LoadRating")
                .master("local")
                .getOrCreate();
        System.out.println("Using Apache Spark v" + spark.version());

        StructType schema = DataTypes.createStructType(new StructField[] {
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

        Dataset<Row> df = spark.read()
                .format("csv")
                .option("header", "true")
                .schema(schema)
                .load("data/ratings.csv");

        var df2 = df.withColumn("datetime", functions.from_unixtime(df.col("timestamp")));
        var splited_datetime = split(df2.col("datetime"),"-");
        df2 = df2
                .withColumn("year", splited_datetime.getItem(0))
                .withColumn("month", splited_datetime.getItem(1))
                .withColumn("day", split(splited_datetime.getItem(2), " ").getItem(0));


        var df_stats_ym = df2.groupBy(df2.col("year"),df2.col("month")).count().orderBy(df2.col("year"),df2.col("month"));
        df_stats_ym.show(1000);

        System.out.println("Excerpt of the dataframe content:");
        df2.show(20);
        System.out.println("Dataframe's schema:");
        df2.printSchema();
        plot_stats_ym(df_stats_ym,"Liczba ocen w kolejnych miesiacach","ratings");

    }


    static void plot_stats_ym(Dataset<Row> df, String title, String label) {
        var labels = df.select(concat(col("year"), lit("-"), col("month"))).as(Encoders.STRING()).collectAsList();
        var x = NumpyUtils.arange(0, labels.size() - 1, 1);
        x = df.select(expr("year+(month-1)/12")).as(Encoders.DOUBLE()).collectAsList();
        var y = df.select("count").as(Encoders.DOUBLE()).collectAsList();
        Plot plt = Plot.create();
        plt.plot().add(x, y).linestyle("-").label(label);
        plt.legend();
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