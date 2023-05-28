package activation

import "github.com/julioguillermo/neuralnetwork/pkg/tensor"

type Linear struct{}

func NewLinear() *Linear {
	return &Linear{}
}

func (linear *Linear) Activate(input tensor.Tensor) (tensor.Tensor, error) {
	return input.Copy(), nil
}

func (linear *Linear) Derive(input tensor.Tensor) (tensor.Tensor, error) {
	return tensor.NewOneTensor(input.GetShape()...), nil
}
