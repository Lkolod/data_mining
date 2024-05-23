import com.github.sh0nk.matplotlib4j.NumpyUtils;
import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonConfig;
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
import java.util.Collections;
import java.util.List;
import java.util.function.Function;
import java.util.stream.IntStream;

import static org.apache.spark.sql.functions.*;

public class LinearRegressionPolynomialFeaturesOrderThree {
    static void processDataset_3nd_order(SparkSession spark, String filename, Function<Double,Double> f_true, int order){

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

        var df_x2 = df.withColumn("X2",expr("X * X")).withColumn("X3",expr("X * X * X"));

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"X", "X2","X3"})
                .setOutputCol("features");

        Dataset<Row> df_transformed = assembler.transform(df_x2);
        //df_transformed.show(5)

        LinearRegression lr = new LinearRegression()
                .setMaxIter(100)
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

        List<Double> X = df_transformed.select("X").as(Encoders.DOUBLE())
                .collectAsList();
        List<Double> Y = df_transformed.select("Y").as(Encoders.DOUBLE())
                .collectAsList();
        plot(X,Y,lrModel,"linear regression xy-005",f_true,order);
    }
    static void plot(List<Double> x, List<Double> y, LinearRegressionModel lrModel, String title, Function<Double, Double> f_true,int order) {
        //Plot plt = Plot.create(PythonConfig.pythonBinPathConfig("C:\\Users\\kolod\\anaconda3\\envs\\pythonProject1\\python.exe"));
        Plot plt = Plot.create();
        plt.plot().add(x, y,"o").label("data");
        double xmin = Collections.min(x);
        double xmax = Collections.max(x);
        var xdelta = 0.05*(xmax-xmin);
        var fx = NumpyUtils.linspace(xmin-xdelta,xmax+xdelta,100);

        List<Double> fy = new ArrayList<>();
        List<Double> to_predict = new ArrayList<>();
        for (Double val : fx) {
            to_predict.clear();
            for (int i = 1; i <= order; i++) {
                to_predict.add(Math.pow(val, i));
            }
            double[] values = to_predict.stream().mapToDouble(Double::doubleValue).toArray();
            DenseVector fx_dense = new DenseVector(values);
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
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("lab3.LoadFunction")
                .master("local")
                .getOrCreate();
        System.out.println("Using Apache Spark v" + spark.version());
        Function<Double, Double> xy2 = x -> -1.5 * x*x + 3*x+4;
        Function<Double, Double> xy3 = x -> -1.5 * x*x + 3*x+4;
        Function<Double, Double> xy4 = x -> -10 * x*x + 500*x-25;
        Function<Double, Double> xy5 = x -> (x + 4) * (x + 1) * (x - 3);

        //processDataset_3nd_order(spark,"data/xy-002.csv",xy2,3);
        //processDataset_3nd_order(spark,"data/xy-003.csv",xy3,3);
        //processDataset_3nd_order(spark,"data/xy-004.csv",xy4,3);
        processDataset_3nd_order(spark,"data/xy-005.csv",xy5,3);

    }
}
