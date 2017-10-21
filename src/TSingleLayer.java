
import ActivationFunction.IActivationFunction;

/**
 * fInputValue, fWeight, fBias, fActivationFunction をセットして利用．
 */
public class TSingleLayer {
    // 入力層の数
    private int fInputNum;
    // 出力層の数
    private int fOutputNum;

    // 重み
    /**
     * 出力層の番号ごとに並べる．
     * fWeight = [ 1to1, 2to1, 3to1, 1to2, 2to2, 3to2, 1to3, 2to3, 3to3 ]
     */
    private double[] fWeight;
    // バイアス
    private double[] fBias;
    // 活性化関数
    private IActivationFunction fActivationFunction;

    /**
     * 入力
     * 前層の出力　＝　現在の入力
     */
    private double[] fInputValue;
    /**
     * 出力
     */
    private double[] fOutputValue;
    /**
     * 出力層に入力する値．
     * fInputToOutputValue = sum( fInputValue[..] * fWeight[..] )
     */
    private double[] fInputToOutputValue;
    /**
     * 誤差逆伝播法による入力層への伝播値
     */
    private double[] fBackToInputValue;
    /**
     * 誤差逆伝播法によって計算された勾配．
     * 各重み，各バイアス項に対して計算される．
     * 値を保存する順序としては，以下のとおりである．
     * fBackPropagationValue = [ w_1_to_1, w_2_to_1, ... , w_fInputNum_to_1, w_1_to_2, ... , w_fInputNum_to_fOutputNum, b_1, .... , b_fOutputNum ]
     */
    private double[] fBackPropagationValue;

    public TSingleLayer(int inputNum, int outputNum, IActivationFunction activationFunction) {
        fInputNum = inputNum;
        fOutputNum = outputNum;
        fWeight = new double[fInputNum * fOutputNum];
        fBias = new double[fOutputNum];
        fActivationFunction = activationFunction;
        fInputValue = new double[fInputNum];
        fOutputValue = new double[fOutputNum];
        fInputToOutputValue = new double[fOutputNum];
        fBackToInputValue = new double[fInputNum];
        fBackPropagationValue = new double[fInputNum * fOutputNum + fOutputNum];
    }

    /**
     * 前層の出力である fInputValue に対して fWeight を掛け合わせて
     * fInputToOutputValue に保存．
     * その後出力層の活性化関数に通し，出力 fOutputValue を計算．
     * @return 出力
     */
    public double[] getOutputValue(){
        double inputValue;
        for(int out = 0; out < fOutputNum; out++) {
            inputValue = fBias[out];
            for (int in = 0; in < fInputNum; in++) {
                int index = out * fInputNum + in;
                inputValue += fInputValue[ in ] * fWeight[ index ];
            }
            fInputToOutputValue[ out ] = inputValue;
            fOutputValue[ out ] = fActivationFunction.forward(inputValue);
        }
        return fOutputValue;
    }

    /**
     * BackPropagation して，fInputNum だけ誤差を伝搬する．
     * fBackPropagationValue に重み，バイアス項の勾配の保存．
     * @param backPropagationValueFromOut ひとつ先の層から得た誤差
     * @return 伝搬値（重みの勾配そのものではないので注意）
     */
    public double[] doBackPropagation(double[] backPropagationValueFromOut){
        assert backPropagationValueFromOut.length == fOutputNum;
        for(int out = 0; out < fOutputNum; out++) {
            /** fBias の勾配*/
            fBackPropagationValue[fInputNum * fOutputNum + out] = fActivationFunction.backward( fOutputValue[out], fInputToOutputValue[out], backPropagationValueFromOut[out] );
        }
        for(int in = 0; in < fInputNum; in++) {
            fBackToInputValue[in] = 0.0;
            for(int out = 0; out < fOutputNum; out++) {
                fBackToInputValue[in] += fWeight[out * fInputNum + in] * fBackPropagationValue[fInputNum * fOutputNum + out];
            }
            for(int out = 0; out < fOutputNum; out++){
                /** fWeight の勾配 */
                fBackPropagationValue[out * fInputNum + in] = fBackPropagationValue[fInputNum * fOutputNum + out] * fInputValue[in];
            }
        }
        return fBackToInputValue;
    }

    /**
     * 前層の出力を fInputValue に保存．
     * @param inputValue 入力
     */
    public void setInputValue(double[] inputValue){
        assert inputValue.length == fInputNum;
        for(int i = 0; i < fInputNum; i++){
            fInputValue[i] = inputValue[i];
        }
    }

    /**
     * 重みをセットする．（ディープコピー）
     * @param weight 重み
     */
    public void setWeight(double[] weight){
        assert weight.length == fInputNum * fOutputNum;
        for(int i = 0; i < weight.length; i++){
            fWeight[i] = weight[i];
        }
    }

    /**
     * バイアス項をセットする．（ディープコピー）
     * @param bias バイアス項
     */
    public void setBias(double[] bias){
        assert bias.length == fOutputNum;
        for(int i = 0; i < fOutputNum; i++){
            fBias[i] = bias[i];
        }
    }

    /**
     * 活性化関数をセット（シャローコピー）
     * @param activationFunction 活性化関数
     */
    public void setActivatioinFunction(IActivationFunction activationFunction){
        fActivationFunction = activationFunction;
    }

    /**
     * 勾配の取得．
     * @return 誤差逆伝播法によって求めた勾配
     */
    public double[] getGradient(){
        return fBackPropagationValue;
    }

    /**
     * 重みの取得
     * @return 重み
     */
    public double[] getWeight() {
        return fWeight;
    }

    /**
     * バイアス項の取得
     * @return バイアス項
     */
    public double[] getBias() {
        return fBias;
    }
}
