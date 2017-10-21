
import ActivationFunction.IActivationFunction;

/**
 * fInputValue, fWeight, fBias, fActivationFunction ���Z�b�g���ė��p�D
 */
public class TSingleLayer {
    // ���͑w�̐�
    private int fInputNum;
    // �o�͑w�̐�
    private int fOutputNum;

    // �d��
    /**
     * �o�͑w�̔ԍ����Ƃɕ��ׂ�D
     * fWeight = [ 1to1, 2to1, 3to1, 1to2, 2to2, 3to2, 1to3, 2to3, 3to3 ]
     */
    private double[] fWeight;
    // �o�C�A�X
    private double[] fBias;
    // �������֐�
    private IActivationFunction fActivationFunction;

    /**
     * ����
     * �O�w�̏o�́@���@���݂̓���
     */
    private double[] fInputValue;
    /**
     * �o��
     */
    private double[] fOutputValue;
    /**
     * �o�͑w�ɓ��͂���l�D
     * fInputToOutputValue = sum( fInputValue[..] * fWeight[..] )
     */
    private double[] fInputToOutputValue;
    /**
     * �덷�t�`�d�@�ɂ����͑w�ւ̓`�d�l
     */
    private double[] fBackToInputValue;
    /**
     * �덷�t�`�d�@�ɂ���Čv�Z���ꂽ���z�D
     * �e�d�݁C�e�o�C�A�X���ɑ΂��Čv�Z�����D
     * �l��ۑ����鏇���Ƃ��ẮC�ȉ��̂Ƃ���ł���D
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
     * �O�w�̏o�͂ł��� fInputValue �ɑ΂��� fWeight ���|�����킹��
     * fInputToOutputValue �ɕۑ��D
     * ���̌�o�͑w�̊������֐��ɒʂ��C�o�� fOutputValue ���v�Z�D
     * @return �o��
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
     * BackPropagation ���āCfInputNum �����덷��`������D
     * fBackPropagationValue �ɏd�݁C�o�C�A�X���̌��z�̕ۑ��D
     * @param backPropagationValueFromOut �ЂƂ�̑w���瓾���덷
     * @return �`���l�i�d�݂̌��z���̂��̂ł͂Ȃ��̂Œ��Ӂj
     */
    public double[] doBackPropagation(double[] backPropagationValueFromOut){
        assert backPropagationValueFromOut.length == fOutputNum;
        for(int out = 0; out < fOutputNum; out++) {
            /** fBias �̌��z*/
            fBackPropagationValue[fInputNum * fOutputNum + out] = fActivationFunction.backward( fOutputValue[out], fInputToOutputValue[out], backPropagationValueFromOut[out] );
        }
        for(int in = 0; in < fInputNum; in++) {
            fBackToInputValue[in] = 0.0;
            for(int out = 0; out < fOutputNum; out++) {
                fBackToInputValue[in] += fWeight[out * fInputNum + in] * fBackPropagationValue[fInputNum * fOutputNum + out];
            }
            for(int out = 0; out < fOutputNum; out++){
                /** fWeight �̌��z */
                fBackPropagationValue[out * fInputNum + in] = fBackPropagationValue[fInputNum * fOutputNum + out] * fInputValue[in];
            }
        }
        return fBackToInputValue;
    }

    /**
     * �O�w�̏o�͂� fInputValue �ɕۑ��D
     * @param inputValue ����
     */
    public void setInputValue(double[] inputValue){
        assert inputValue.length == fInputNum;
        for(int i = 0; i < fInputNum; i++){
            fInputValue[i] = inputValue[i];
        }
    }

    /**
     * �d�݂��Z�b�g����D�i�f�B�[�v�R�s�[�j
     * @param weight �d��
     */
    public void setWeight(double[] weight){
        assert weight.length == fInputNum * fOutputNum;
        for(int i = 0; i < weight.length; i++){
            fWeight[i] = weight[i];
        }
    }

    /**
     * �o�C�A�X�����Z�b�g����D�i�f�B�[�v�R�s�[�j
     * @param bias �o�C�A�X��
     */
    public void setBias(double[] bias){
        assert bias.length == fOutputNum;
        for(int i = 0; i < fOutputNum; i++){
            fBias[i] = bias[i];
        }
    }

    /**
     * �������֐����Z�b�g�i�V�����[�R�s�[�j
     * @param activationFunction �������֐�
     */
    public void setActivatioinFunction(IActivationFunction activationFunction){
        fActivationFunction = activationFunction;
    }

    /**
     * ���z�̎擾�D
     * @return �덷�t�`�d�@�ɂ���ċ��߂����z
     */
    public double[] getGradient(){
        return fBackPropagationValue;
    }

    /**
     * �d�݂̎擾
     * @return �d��
     */
    public double[] getWeight() {
        return fWeight;
    }

    /**
     * �o�C�A�X���̎擾
     * @return �o�C�A�X��
     */
    public double[] getBias() {
        return fBias;
    }
}
