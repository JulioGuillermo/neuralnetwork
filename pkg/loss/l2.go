package loss

import "github.com/julioguillermo/neuralnetwork/pkg/tensor"

func L2(output, target tensor.Tensor) (tensor.Tensor, error) {
	loss := target.Copy()

	err := loss.SubTensor(output)
	if err != nil {
		return nil, err
	}

	sign := loss.Sign()

	err = loss.PowNumber(2)
	if err != nil {
		return nil, err
	}

	err = loss.MulTensor(sign)
	if err != nil {
		return nil, err
	}
	return loss, nil
}
