package loss

import "github.com/julioguillermo/neuralnetwork/pkg/tensor"

func L3(output, target tensor.Tensor) (tensor.Tensor, error) {
	loss := target.Copy()

	err := loss.SubTensor(output)
	if err != nil {
		return nil, err
	}

	err = loss.PowNumber(3)
	if err != nil {
		return nil, err
	}

	return loss, nil
}
