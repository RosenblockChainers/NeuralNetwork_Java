
import ActivationFunction.IActivationFunction;
import ActivationFunction.TSigmoidFunction;

import java.util.Random;

public class TMain {
    public static void main(String[] args) {
        IActivationFunction function = new TSigmoidFunction();

        int[] nodeNum = new int[]{2, 3, 4, 5, 6, 5, 4, 3, 2};
        TNeuralNetwork neuralNetwork = new TNeuralNetwork(nodeNum, function);
        int bNum = 0, wNum = 0;
        for(int i = 0; i < nodeNum.length - 1; i++){
            bNum += nodeNum[i + 1];
            wNum += nodeNum[i] * nodeNum[i + 1];
        }
        double[] bias = new double[bNum];
        double[] weight = new double[wNum];
        for(int i = 0; i < bias.length; i++){
            bias[i] = 0.1 * i;
        }
        for(int i = 0; i < weight.length; i++){
            weight[i] = 0.01 * i;
        }
        int biasIdx = 0, weightIdx = 0;
        for(int i = 0; i < neuralNetwork.getLayerSettingNum(); i++){
            int biasNum = neuralNetwork.getLayerSetting(i).getBias().length;
            int weightNum = neuralNetwork.getLayerSetting(i).getWeight().length;
            for(int j = 0; j < biasNum; j++) {
                neuralNetwork.getLayerSetting(i).getBias()[j] = bias[biasIdx];
                biasIdx++;
            }
            for(int j = 0; j < weightNum; j++){
                neuralNetwork.getLayerSetting(i).getWeight()[j] = weight[weightIdx];
                weightIdx++;
            }
        }

        double[] input = new double[]{1.2, 1.4};
        System.out.print("test: output: ");
        double[] o = neuralNetwork.getOutputValue(input);
        for(int i = 0; i < 2; i++) {
            System.out.print(o[i] + ",");
        }
        System.out.println();
        Random random = new Random();
        double out[];
        double[] backInit = new double[]{1.0, 1.0};
        long start, end;
        System.out.println("start");
        start = System.currentTimeMillis();
        for(int num = 0; num < 100000; num++) {
            input[0] = random.nextDouble();
            input[1] = random.nextDouble();
            out = neuralNetwork.getOutputValue(input);
            neuralNetwork.doBackPropagation(backInit);
        }
        end = System.currentTimeMillis();
        System.out.println("time: " + (end - start) + "[ms]");
        System.out.println("end");
//        neuralNetwork.doBackPropagation(backInit);
//        double[] testtest = neuralNetwork.getGradient();
//        double[] testweight = neuralNetwork.getGradientNoBias();
//        System.out.println("GradientNoBias");
//        for(int i = 0; i < testweight.length; i++){
//            System.out.println(i + ": " + testweight[i]);
//        }
//        System.out.println("Gradient");
//        for(int i = 0; i < testtest.length; i++){
//            System.out.println(i + ":" + testtest[i]);
//        }
    }
}
