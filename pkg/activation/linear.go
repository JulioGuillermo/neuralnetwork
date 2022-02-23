package activation

import "github.com/JulioGuillermo/neuralnetwork/pkg/tensor"

type Linear struct{}

func NewLinear() *Linear {
	return &Linear{}
}

func (self *Linear) Activate(input tensor.Tensor) (tensor.Tensor, error) {
	return input.Copy(), nil
}

func (self *Linear) Derive(input tensor.Tensor) (tensor.Tensor, error) {
	return tensor.NewOneTensor(input.GetShape()...), nil
}
