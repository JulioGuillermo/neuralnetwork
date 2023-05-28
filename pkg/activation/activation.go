package activation

import "github.com/julioguillermo/neuralnetwork/pkg/tensor"

type Activation interface {
	Activate(tensor.Tensor) (tensor.Tensor, error)
	Derive(tensor.Tensor) (tensor.Tensor, error)
}
