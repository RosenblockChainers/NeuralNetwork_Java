
import ActivationFunction.IActivationFunction;

public class TNeuralNetwork {
    /**
     * �w�̐�
     */
    private int fLayerNum;
    /**
     * �e�w���Ƃ̃m�[�h��
     */
    private int[] fNodeNum;
    /**
     * �e�w���Ƃ̏d�݂Ƃ�
     */
    private TSingleLayer[] fLayerSetting;

    /**
     * �d�݁C�o�C�A�X���̐��������z��ۑ�����D
     * ���Ԃ́C�j���[�����l�b�g�̓��͑w���珇�ɕۑ�����D
     * fBackPropagation �� �ۑ�����D
     * ���̂��߁CW1,B1,W2,B2 ....�ƂȂ�D
     * W, B �͊e�w���ƂɈȉ��̂悤�ɕۑ�����Ă���D�i�ڂ����� TSingleLayer �ցj
     * W = w11, w21, w31, ......
     * B = b1, b2, ........
     */
    private double[] fGradient;
    /**
     * �o�C�A�X����0�Ƃ����Ƃ��̌��z�D
     * �d�݂̌��z�݂̂��ۑ������D
     */
    private double[] fGradientNoBias;

    /**
     * �S�Ă̑w���������D
     * �S�Ẵm�[�h�̊����֐��𓯂��֐��ŏ�����
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
     * ���͒l�������Ɏ��C�o�͒l��Ԃ��D
     * �Ō�̑w�̏o�͂��V�����[�R�s�[�ŕԂ��D
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
     * �o�b�N�v���p�Q�[�V����������ɌĂяo���āC���z���擾
     * @return ���z
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
     * �e�w���擾�D
     * ��������d�݂Ƃ�������������D
     * �������֐����������珑��������D
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
