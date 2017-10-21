package ActivationFunction;

/**
 * Created by Yusuke on 2017/04/06.
 */
public interface IActivationFunction {
    abstract double forward(double input);

    /**
     * @param output ���͑w�� Output �Ȃ̂Ŏ������ fInputValue�������ɓ��͂���D
     * @param input ���͑w�� Input �Ȃ̂Ŏ�����͂ЂƂO�̑w�� fO
     * @param valueFromOut
     * @return
     */
    abstract double backward(double output, double input, double valueFromOut);
}
