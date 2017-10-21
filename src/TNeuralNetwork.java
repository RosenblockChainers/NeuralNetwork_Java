
import ActivationFunction.IActivationFunction;

public class TNeuralNetwork {
    /**
     * 層の数
     */
    private int fLayerNum;
    /**
     * 各層ごとのノード数
     */
    private int[] fNodeNum;
    /**
     * 各層ごとの重みとか
     */
    private TSingleLayer[] fLayerSetting;

    /**
     * 重み，バイアス項の数だけ勾配を保存する．
     * 順番は，ニューラルネットの入力層から順に保存する．
     * fBackPropagation を 保存する．
     * そのため，W1,B1,W2,B2 ....となる．
     * W, B は各層ごとに以下のように保存されている．（詳しくは TSingleLayer へ）
     * W = w11, w21, w31, ......
     * B = b1, b2, ........
     */
    private double[] fGradient;
    /**
     * バイアス項を0としたときの勾配．
     * 重みの勾配のみが保存される．
     */
    private double[] fGradientNoBias;

    /**
     * 全ての層を初期化．
     * 全てのノードの活性関数を同じ関数で初期化
     * @param nodeNum
     * @param activationFunction
     */
    public TNeuralNetwork(int[] nodeNum, IActivationFunction activationFunction){
        fLayerNum = nodeNum.length;
        fNodeNum = new int[fLayerNum];
        fLayerSetting = new TSingleLayer[fLayerNum - 1];
        for(int i = 0; i < fLayerNum; i++){
            fNodeNum[i] = nodeNum[i];
        }
        for(int i = 0; i < fLayerSetting.length; i++){
            fLayerSetting[i] = new TSingleLayer(fNodeNum[i], fNodeNum[i + 1], activationFunction);
        }

        int allBranchNum = 0;
        int allBranchNumNoBias = 0;
        for(int i = 0; i < nodeNum.length - 1; i++){
            allBranchNum += nodeNum[i] * nodeNum[i + 1] + nodeNum[i + 1];
            allBranchNumNoBias += nodeNum[i] * nodeNum[i + 1];
        }
        fGradient = new double[allBranchNum];
        fGradientNoBias = new double[allBranchNumNoBias];
    }

    /**
     * 入力値を引数に取り，出力値を返す．
     * 最後の層の出力をシャローコピーで返す．
     * @param inputValue
     * @return
     */
    public double[] getOutputValue(double[] inputValue){
        double[] outputValue;
        outputValue = inputValue;
        for(int i = 0; i < fLayerSetting.length; i++){
            fLayerSetting[i].setInputValue(outputValue);
            outputValue = fLayerSetting[i].getOutputValue();
        }
        return outputValue;
    }

    public double[] doBackPropagation(double[] initValue){
        double[] backPropagationValue;
        backPropagationValue = initValue;
        for(int i = fLayerSetting.length - 1; i > -1; i--){
            backPropagationValue = fLayerSetting[i].doBackPropagation(backPropagationValue);
        }
        return backPropagationValue;
    }

    /**
     * バックプロパゲーションした後に呼び出して，勾配を取得
     * @return 勾配
     */
    public double[] getGradient(){
        int gradientIndex = 0;
        for(int i = 0; i < fLayerSetting.length; i++){
            for(int j = 0; j < fLayerSetting[i].getGradient().length; j++){
                fGradient[gradientIndex] = fLayerSetting[i].getGradient()[j];
                gradientIndex++;
            }
        }
        return fGradient;
    }
    public double[] getGradientNoBias(){
        int gradientIndex = 0;
        for(int i = 0; i < fLayerSetting.length; i++){
            for(int j = 0; j < fLayerSetting[i].getWeight().length; j++){
                fGradientNoBias[gradientIndex] = fLayerSetting[i].getGradient()[j];
                gradientIndex++;
            }
        }
        return fGradientNoBias;
    }

    /**
     * 各層を取得．
     * ここから重みとかを初期化する．
     * 活性化関数もここから書き換える．
     * @param idx
     * @return
     */
    public TSingleLayer getLayerSetting(int idx) {
        return fLayerSetting[idx];
    }

    public int getLayerSettingNum() {
        return fLayerSetting.length;
    }
}
