package ActivationFunction;

/**
 * Created by Yusuke on 2017/04/06.
 */
public interface IActivationFunction {
    abstract double forward(double input);

    /**
     * @param output “ü—Í‘w‚Ì Output ‚È‚Ì‚ÅÀ‘•ã‚Í fInputValue‚ğˆø”‚É“ü—Í‚·‚éD
     * @param input “ü—Í‘w‚Ì Input ‚È‚Ì‚ÅÀ‘•ã‚Í‚Ğ‚Æ‚Â‘O‚Ì‘w‚Ì fO
     * @param valueFromOut
     * @return
     */
    abstract double backward(double output, double input, double valueFromOut);
}
