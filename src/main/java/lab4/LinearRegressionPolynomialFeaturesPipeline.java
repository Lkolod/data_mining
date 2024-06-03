package lab4;

import com.github.sh0nk.matplotlib4j.NumpyUtils;
import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonConfig;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.PolynomialExpansion;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

import static java.util.stream.Collectors.toList;
import static org.apache.spark.sql.functions.expr;

public class LinearRegressionPolynomialFeaturesPipeline {
        static void processDataset(SparkSession spark, String filename, Function<Double, Double> f_true, int degree) {

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
                    .load(filename);

            df = df.orderBy(org.apache.spark.sql.functions.rand(3));
            long rowsCount = df.count();
            int trainCount = (int)(rowsCount*.7);
            var df_train = df.select("*").limit(trainCount);
            var df_test = df.select("*").offset(trainCount);
            System.out.println(df_train.count());
            System.out.println(df_test.count());

//            var dfs = df.randomSplit(new double[]{0.7,0.3}, 3);
//            var df_train = dfs[0];
//            var df_test = dfs[1];

            VectorAssembler vectorAssembler = new VectorAssembler()
                    .setInputCols(new String[]{"X"})
                    .setOutputCol("features");

            //Dataset<Row> df_transformed = vectorAssembler.transform(df);
            //df_transformed.show(5)
            PolynomialExpansion polyExpansion = new PolynomialExpansion()
                    .setInputCol("features")
                    .setOutputCol("polyFeatures")
                    .setDegree(degree);

            //df_transformed = polyExpansion.transform(df_transformed);

            LinearRegression lr = new LinearRegression()
                    .setMaxIter(100)
                    .setRegParam(0.3)
                    .setElasticNetParam(0.8)
                    .setFeaturesCol("polyFeatures")
                    .setLabelCol("Y");


            Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{vectorAssembler, polyExpansion, lr});
            PipelineModel model = pipeline.fit(df_train);
            LinearRegressionModel lrModel = (LinearRegressionModel) model.stages()[2];

            //task 4.4
            var df_test_prediction = model.transform(df_test);
            RegressionEvaluator evaluator = new RegressionEvaluator()
                    .setLabelCol("Y")
                    .setPredictionCol("prediction")
                    .setMetricName("rmse"); // or any other evaluation metric
            double rmse = evaluator.evaluate(df_test_prediction);
            System.out.println("RMSE: " + rmse);
            evaluator.setMetricName("r2");
            double r2 = evaluator.evaluate(df_test_prediction);
            System.out.println("R^2: " + r2);

            System.out.println(lrModel.coefficients());
            System.out.println(lrModel.intercept());

            LinearRegressionTrainingSummary trainingSummary = lrModel.summary();
//            System.out.println("numIterations: " + trainingSummary.totalIterations());
//            System.out.println("objectiveHistory: " + Vectors.dense(trainingSummary.objectiveHistory()));
//            trainingSummary.residuals().show(100);
//            System.out.println("MSE: " + trainingSummary.meanSquaredError());
//            System.out.println("RMSE: " + trainingSummary.rootMeanSquaredError());
//            System.out.println("MAE: " + trainingSummary.meanAbsoluteError());
//            System.out.println("r2: " + trainingSummary.r2());

            List<Double> X = df_train.select("X").as(Encoders.DOUBLE())
                    .collectAsList();
            List<Double> Y = df_train.select("Y").as(Encoders.DOUBLE())
                    .collectAsList();

            plot(X, Y, model, spark, String.format("Linear regression: %s (train data)",filename), f_true);
            //plot(X, Y, model, spark, "regresion", f_true);
            var x = df_test.select("X").as(Encoders.DOUBLE()).collectAsList();
            var y = df_test.select("Y").as(Encoders.DOUBLE()).collectAsList();
            plot(x,y,model,spark,String.format("Linear regression: %s (test data)",filename),f_true);
        }

        static void plot(List<Double>x, List<Double> y, PipelineModel pipelineModel, SparkSession spark, String title, Function<Double,Double> f_true) {
            //Plot plt = Plot.create(PythonConfig.pythonBinPathConfig("C:\\Users\\kolod\\anaconda3\\envs\\pythonProject1\\python.exe"));
            Plot plt = Plot.create();
            plt.plot().add(x, y, "o").label("data");
            double xmin = Collections.min(x);
            double xmax = Collections.max(x);
            var xdelta = 0.05 * (xmax - xmin);
            var fx = NumpyUtils.linspace(xmin - xdelta, xmax + xdelta, 100);

            List<Row> rows = fx.stream()
                    .map(d -> RowFactory.create(d))
                    .collect(toList());

            StructType schema = new StructType().add("X", "double");
            Dataset<Row> df_test =  spark.createDataFrame(rows,schema);

            //df_test.show(5);
            //df_test.printSchema();
            Dataset<Row> df_pred = pipelineModel.transform(df_test);
            df_pred.show(5);
            df_pred.printSchema();
            List<Double> f_x = df_pred.select("X").as(Encoders.DOUBLE()).collectAsList();
            List<Double> f_y = df_pred.select("prediction").as(Encoders.DOUBLE()).collectAsList();
            plt.plot().add(f_x, f_y).color("r").label("pred");
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
            //for (Double value : f_x) {
            //    System.out.println(value);
            //}
        }

        public static void main(String[] args) {
            SparkSession spark = SparkSession.builder()
                    .appName("lab3.LoadFunction")
                    .master("local")
                    .getOrCreate();
            System.out.println("Using Apache Spark v" + spark.version());
            Function<Double, Double> xy2 = x -> -1.5 * x * x + 3 * x + 4;
            Function<Double, Double> xy3 = x -> -1.5 * x * x + 3 * x + 4;
            Function<Double, Double> xy4 = x -> -10 * x * x + 500 * x - 25;
            Function<Double, Double> xy5 = x -> (x + 4) * (x + 1) * (x - 3);
            //processDataset_3nd_order(spark,"data/xy-002.csv",xy2,3);
            //processDataset_3nd_order(spark,"data/xy-003.csv",xy3,3);
            //processDataset_3nd_order(spark,"data/xy-004.csv",xy4,3);
            processDataset(spark,"data/xy-003.csv",xy3,3);

        }
}