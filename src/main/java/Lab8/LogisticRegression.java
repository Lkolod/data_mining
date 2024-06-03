package Lab8;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class LogisticRegression {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("LogisticRegressionOnExam")
                .master("local")
                .getOrCreate();
        System.out.println("Using Apache Spark v" + spark.version());

        Dataset<Row> df = loadData(spark, "src/main/resources/Lab8/egzamin-cpp.csv");
        Dataset<Row> df2 = loadData(spark, "src/main/resources/Lab8/grid.csv");

        //TODO poprawid loadData

    }
    public static Dataset<Row> loadData(SparkSession spark,String filePath){

        Dataset<Row> df = spark.read()
                .format("csv")
                .option("header", "true")
                .load(filePath);
        df.show();
        df.printSchema();
        return df;
    }
}
