package loss

import "github.com/julioguillermo/neuralnetwork/pkg/tensor"

func L1(output, target tensor.Tensor) (tensor.Tensor, error) {
	loss := target.Copy()
	err := loss.SubTensor(output)
	return loss, err
}
