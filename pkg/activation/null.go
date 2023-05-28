package activation

import "github.com/julioguillermo/neuralnetwork/pkg/tensor"

type Null struct{}

func NewNull() *Null {
	return &Null{}
}

func (null *Null) Activate(input tensor.Tensor) (tensor.Tensor, error) {
	return input.Copy(), nil
}

func (null *Null) Derive(input tensor.Tensor) (tensor.Tensor, error) {
	return input.Copy(), nil
}

var ActNull = NewNull()
