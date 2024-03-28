import com.github.sh0nk.matplotlib4j.NumpyUtils;
import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class LoadFunction {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("LoadFunction")
                .master("local")
                .getOrCreate();
        System.out.println("Using Apache Spark v" + spark.version());

        StructType schema = DataTypes.createStructType(new StructField[]{
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
                .load("data/xy-001.csv");

        Dataset<Row> df_2 = spark.read()
                .format("csv")
                .option("header", "true")
                .schema(schema)
                .load("data/xy-002.csv");

        Dataset<Row> df_3 = spark.read()
                .format("csv")
                .option("header", "true")
                .schema(schema)
                .load("data/xy-003.csv");

        Dataset<Row> df_4 = spark.read()
                .format("csv")
                .option("header", "true")
                .schema(schema)
                .load("data/xy-004.csv");

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"X"})
                .setOutputCol("features");

        Dataset<Row> df_transformed = assembler.transform(df);
        Dataset<Row> df_transformed_2 = assembler.transform(df_2);
        Dataset<Row> df_transformed_3 = assembler.transform(df_3);
        Dataset<Row> df_transformed_4 = assembler.transform(df_4);

        //task 1 end
        //df_transformed.show();

        //task 1.2
        //task 1.2.1
        LinearRegression lr = new LinearRegression()
                //task 1.4
                .setMaxIter(10)
                //10,20,50,100 here change it
                .setRegParam(0.3)
                //.setRegParam(0.0)
                // numIterations: 0 if (0.0)
                .setElasticNetParam(0.8)
                //.setElasticNetParam(0.0)
                .setFeaturesCol("features")
                .setLabelCol("Y");

        // Fit the model.
        LinearRegressionModel lrModel = lr.fit(df_transformed);
        LinearRegressionModel lrModel_2 = lr.fit(df_transformed_2);
        LinearRegressionModel lrModel_3 = lr.fit(df_transformed_3);
        LinearRegressionModel lrModel_4 = lr.fit(df_transformed_4);

        //task 1.2.2
        System.out.println(lrModel.coefficients());
        System.out.println(lrModel.intercept());

        //task 1.2.3
        LinearRegressionTrainingSummary trainingSummary = lrModel.summary();
        System.out.println("numIterations: " + trainingSummary.totalIterations());
        System.out.println("objectiveHistory: " + Vectors.dense(trainingSummary.objectiveHistory()));
        trainingSummary.residuals().show(100);
        System.out.println("MSE: " + trainingSummary.meanSquaredError());
        System.out.println("RMSE: " + trainingSummary.rootMeanSquaredError());
        System.out.println("MAE: " + trainingSummary.meanAbsoluteError());
        System.out.println("r2: " + trainingSummary.r2());

        //1.2.5
        var summary = Arrays.stream(trainingSummary.objectiveHistory())
                .boxed()
                .collect(Collectors.toList());

        //plotObjectiveHistory(summary);

        //1.3
        List<Double> X = df_transformed.select("X").as(Encoders.DOUBLE())
                .collectAsList();
        List<Double> Y = df_transformed.select("Y").as(Encoders.DOUBLE())
                .collectAsList();

        List<Double> X2 = df_transformed_2.select("X").as(Encoders.DOUBLE())
                .collectAsList();
        List<Double> Y2 = df_transformed_2.select("Y").as(Encoders.DOUBLE())
                .collectAsList();

        List<Double> X3 = df_transformed_3.select("X").as(Encoders.DOUBLE())
                .collectAsList();
        List<Double> Y3 = df_transformed_3.select("Y").as(Encoders.DOUBLE())
                .collectAsList();

        List<Double> X4 = df_transformed_4.select("X").as(Encoders.DOUBLE())
                .collectAsList();
        List<Double> Y4 = df_transformed_4.select("Y").as(Encoders.DOUBLE())
                .collectAsList();

        Function<Double, Double> xy1 = x -> 2.37 * x + 7;
        Function<Double, Double> xy2 = x -> -1.5 * x*x + 3*x+4;
        Function<Double, Double> xy3 = x -> -1.5 * x*x + 3*x+4;
        Function<Double, Double> xy4 = x -> -10 * x*x + 500*x-25;

        plot(X,Y,lrModel,"linear regression",xy1);

        //task 2.1
        //plot(X2,Y2,lrModel_2,"linear regression",xy2);
        //plot(X3,Y3,lrModel_3,"linear regression",xy3);
        //plot(X4,Y4,lrModel_4,"linear regression",xy4);
        //task 2.2
        LinearRegressionTrainingSummary trainingSummary_2 = lrModel_2.summary();
        LinearRegressionTrainingSummary trainingSummary_3 = lrModel_3.summary();
        LinearRegressionTrainingSummary trainingSummary_4 = lrModel_4.summary();

        List<String> metricsTable = new ArrayList<>();
        metricsTable.add("Metric\tModel 2\tModel 3\tModel 4");
        metricsTable.add("--------------------------------------------------");
        metricsTable.add("Coefficient of Determination (R-squared)\t" +
                trainingSummary_2.r2() + "\t" +
                trainingSummary_3.r2() + "\t" +
                trainingSummary_4.r2());
        metricsTable.add("Root Mean Squared Error (RMSE)\t" +
                trainingSummary_2.rootMeanSquaredError() + "\t" +
                trainingSummary_3.rootMeanSquaredError() + "\t" +
                trainingSummary_4.rootMeanSquaredError());
        metricsTable.add("Coefficients\t" +
                lrModel_2.coefficients() + "\t" +
                lrModel_3.coefficients() + "\t" +
                lrModel_4.coefficients());
        metricsTable.add("Intercept\t" +
                lrModel_2.intercept() + "\t" +
                lrModel_3.intercept() + "\t" +
                lrModel_4.intercept());
        metricsTable.add("numIterations\t" +
                trainingSummary_2.totalIterations() + "\t" +
                trainingSummary_3.totalIterations() + "\t" +
                trainingSummary_4.totalIterations());
        metricsTable.add("objectiveHistory\t" +
                Vectors.dense(trainingSummary_2.objectiveHistory()) + "\t" +
                Vectors.dense(trainingSummary_3.objectiveHistory()) + "\t" +
                Vectors.dense(trainingSummary_4.objectiveHistory()));
        metricsTable.add("MSE\t" +
                trainingSummary_2.meanSquaredError() + "\t" +
                trainingSummary_3.meanSquaredError() + "\t" +
                trainingSummary_4.meanSquaredError());
        metricsTable.add("MAE\t" +
                trainingSummary_2.meanAbsoluteError() + "\t" +
                trainingSummary_3.meanAbsoluteError() + "\t" +
                trainingSummary_4.meanAbsoluteError());

        for (String row : metricsTable) {
            System.out.println(row);
        }
    }

    static void plotObjectiveHistory(List<Double> lossHistory) {
        var x = IntStream.range(0, lossHistory.size()).mapToDouble(d -> d).boxed().toList();
        Plot plt = Plot.create();
        plt.plot().add(x, lossHistory).label("loss");
        plt.xlabel("Iteration");
        plt.ylabel("Loss");
        plt.title("Loss history");
        plt.legend();
        try {
            plt.show();
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (PythonExecutionException e) {
            throw new RuntimeException(e);
        }
    }
    

    static void plot(List<Double> x, List<Double> y, LinearRegressionModel lrModel, String title, Function<Double, Double> f_true) {
        Plot plt = Plot.create();
        plt.plot().add(x, y,"o").label("data");
        double xmin = Collections.min(x);
        double xmax = Collections.max(x);
        var xdelta = 0.05*(xmax-xmin);
        var fx = NumpyUtils.linspace(xmin-xdelta,xmax+xdelta,100);
        //var fx_dense = new DenseVector(fx.stream().mapToDouble(Double::doubleValue).toArray());
        List<Double> fy = new ArrayList<>();
        for (Double val : fx) {
            DenseVector fx_dense = new DenseVector(new double[]{val});
            double prediction = lrModel.predict(fx_dense);
            fy.add(prediction);
        }

        plt.plot().add(fx, fy).color("r").label("pred");
        if (f_true != null) {
            List<Double> fy_true = new ArrayList<>();
            for (Double _x : fx) {
                double result = f_true.apply(_x);
                fy_true.add(result);
            }
            plt.plot().add(fx, fy_true).color("g").linestyle("--").label("$f_{true}$");

        }
        plt.title(title);
        plt.legend();
        try {
            plt.show();
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (PythonExecutionException e) {
            throw new RuntimeException(e);
        }
    }
}