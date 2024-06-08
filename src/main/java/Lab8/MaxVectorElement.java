package Lab8;

import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.api.java.UDF1;

public class MaxVectorElement implements UDF1<Vector, Double> {
    @Override
    public Double call(Vector vector) throws Exception {
        return vector.apply(vector.argmax());
    }
}
