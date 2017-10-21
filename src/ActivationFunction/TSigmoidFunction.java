package ActivationFunction;

/**
 * Created by Yusuke on 2017/04/06.
 */
public class TSigmoidFunction implements IActivationFunction {
    public double forward(double input){
        return 1.0 / (1.0 + Math.exp(-input));
    }
    public double backward(double output, double input, double valueFromOut){
        return valueFromOut * (1.0 - output) * output;
    }
}
