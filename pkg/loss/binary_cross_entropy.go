package loss

import (
	"math"

	"github.com/julioguillermo/neuralnetwork/pkg/tensor"
)

func BinaryCrossEntropy(output, target tensor.Tensor) (tensor.Tensor, error) {
	loss := target.Copy()

	err := loss.SubTensor(output)
	if err != nil {
		return nil, err
	}

	sign := loss.Sign()

	err = loss.Run(func(o float64, i int) (float64, error) {
		return 1.0 / (1.0 + math.Exp(o)), nil
	})
	if err != nil {
		return nil, err
	}

	err = loss.MulTensor(sign)
	if err != nil {
		return nil, err
	}
	return loss, nil
}
