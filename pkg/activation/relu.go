package activation

import (
	"github.com/julioguillermo/neuralnetwork/pkg/tensor"
)

type Relu struct{}

func NewRelu() *Relu {
	return &Relu{}
}

func (relu *Relu) act(x float64, i int) (float64, error) {
	if x > 0 {
		return x, nil
	}
	return 0, nil
}

func (relu *Relu) der(x float64, i int) (float64, error) {
	if x > 0 {
		return 1, nil
	}
	return 0, nil
}

func (relu *Relu) Activate(input tensor.Tensor) (tensor.Tensor, error) {
	ret := input.Copy()
	e := ret.Run(relu.act)
	return ret, e
}

func (relu *Relu) Derive(input tensor.Tensor) (tensor.Tensor, error) {
	ret := input.Copy()
	e := ret.Run(relu.der)
	return ret, e
}
