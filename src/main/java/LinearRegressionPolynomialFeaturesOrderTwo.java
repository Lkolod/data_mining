import com.github.sh0nk.matplotlib4j.NumpyUtils;
import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.io.IOException;
import java.util.List;
import java.util.function.Function;
import java.util.stream.IntStream;

import static org.apache.spark.sql.functions.*;

public class LinearRegressionPolynomialFeaturesOrderTwo {
    static void processDataset(SparkSession spark, String filename, Function<Double,Double> f_true){

        StructType schema = DataTypes.createStructType(new StructField[] {
                DataTypes.createStructField(
                        "X",
                        DataTypes.DoubleType,
                        true),
                DataTypes.createStructField(
                        "Y",
                        DataTypes.DoubleType,
                        true),
        });

        Dataset<Row> df = spark.read()
                .format("csv")
                .option("header", "true")
                .schema(schema)
                .load(filename);

        var df_x2 = df.withColumn("X2",expr("X * X"));

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"X"})
                .setInputCols(new String[]{"X2"})
                .setOutputCol("features");

        Dataset<Row> df_transformed = assembler.transform(df_x2);

        LinearRegression lr = new LinearRegression()
                .setMaxIter(10)
                .setRegParam(0.3)
                .setElasticNetParam(0.8)
                .setFeaturesCol("features")
                .setLabelCol("Y");

        LinearRegressionModel lrModel = lr.fit(df_transformed);

        System.out.println(lrModel.coefficients());
        System.out.println(lrModel.intercept());

        LinearRegressionTrainingSummary trainingSummary = lrModel.summary();
        System.out.println("numIterations: " + trainingSummary.totalIterations());
        System.out.println("objectiveHistory: " + Vectors.dense(trainingSummary.objectiveHistory()));
        trainingSummary.residuals().show(100);
        System.out.println("MSE: " + trainingSummary.meanSquaredError());
        System.out.println("RMSE: " + trainingSummary.rootMeanSquaredError());
        System.out.println("MAE: " + trainingSummary.meanAbsoluteError());
        System.out.println("r2: " + trainingSummary.r2());

        var summarry = trainingSummary.objectiveHistory();
    }
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("LoadFunction")
                .master("local")
                .getOrCreate();
        System.out.println("Using Apache Spark v" + spark.version());
    }


    }
