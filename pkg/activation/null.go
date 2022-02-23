package activation

import "github.com/JulioGuillermo/neuralnetwork/pkg/tensor"

type Null struct{}

func NewNull() *Null {
	return &Null{}
}

func (self *Null) Activate(input tensor.Tensor) (tensor.Tensor, error) {
	return input.Copy(), nil
}

func (self *Null) Derive(input tensor.Tensor) (tensor.Tensor, error) {
	return input.Copy(), nil
}

var ActNull = NewNull()
