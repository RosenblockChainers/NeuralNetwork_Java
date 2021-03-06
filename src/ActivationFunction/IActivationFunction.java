package ActivationFunction;

/**
 * Created by Yusuke on 2017/04/06.
 */
public interface IActivationFunction {
    abstract double forward(double input);

    /**
     * @param output 入力層の Output なので実装上は fInputValueを引数に入力する．
     * @param input 入力層の Input なので実装上はひとつ前の層の fO
     * @param valueFromOut
     * @return
     */
    abstract double backward(double output, double input, double valueFromOut);
}
