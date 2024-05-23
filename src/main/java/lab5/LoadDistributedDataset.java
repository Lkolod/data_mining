package lab5;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class LoadDistributedDataset {

    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("LoadDatasetDistributed")
                .master("spark://172.22.0.2:7077")
                .getOrCreate();

        Dataset<Row> df = spark.read().format("csv")
                .option("header", "true")
                .option("inferschema","true")
                .load("data/owid-energy-data.csv.gz");

        df.show(5); // To raczej zakomentuj...
        df=df.select("country","year","population","electricity_demand").where("country like \'Po%\' AND year >= 2000");
        df.show(5);
        df.printSchema();


    }
}