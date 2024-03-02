import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import static org.apache.spark.sql.functions.*;

public class LoadMovies {
    public static void main(String[] args) {
                SparkSession spark = SparkSession.builder()
                        .appName("Load movies")
                        .master("local")
                        .getOrCreate();
                System.out.println("Using Apache Spark v" + spark.version());

                StructType schema = DataTypes.createStructType(new StructField[] {
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

                var df2 = df
                        .withColumn("title2", regexp_extract(df.col("title"), "^(.+)\\((\\d{4})\\)$", 1))
                        .withColumn("year", regexp_extract(df.col("title"), "^(.+)\\((\\d{4})\\)$", 2))
                        .drop("title")
                        .withColumnRenamed("title2", "title");

                var df_transformed = df2
                        .withColumn("genres_array", split(df.col("genres"), "\\|"));

                var df_exploded = df2
                        .withColumn("genres_array", split(df.col("genres"), "\\|"))
                        .withColumn("genre", explode(col("genres_array")))
                        .drop("genres_array")
                        .drop("genres");

                df_exploded.select("genre").distinct().show();

                var genreList = df_exploded.select("genre").distinct().as(Encoders.STRING()).collectAsList();

                var df_multigenre = df_transformed;
                for(var s:genreList){
                    if(s.equals("(no genres listed)"))continue;
                        df_multigenre=df_multigenre.withColumn(s,array_contains(col("genres_array"),s));
        }
                System.out.println("Excerpt of the dataframe content:");
                df_multigenre.show(20);
                System.out.println("Dataframe's schema:");
                df_multigenre.printSchema();
            }

        }

