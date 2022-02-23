package activation

import (
	"github.com/JulioGuillermo/neuralnetwork/pkg/tensor"
)

type Relu struct{}

func NewRelu() *Relu {
	return &Relu{}
}

func (self *Relu) act(x float64, i int) (float64, error) {
	if x > 0 {
		return x, nil
	}
	return 0, nil
}

func (self *Relu) der(x float64, i int) (float64, error) {
	if x > 0 {
		return 1, nil
	}
	return 0, nil
}

func (self *Relu) Activate(input tensor.Tensor) (tensor.Tensor, error) {
	ret := input.Copy()
	e := ret.Run(self.act)
	return ret, e
}

func (self *Relu) Derive(input tensor.Tensor) (tensor.Tensor, error) {
	ret := input.Copy()
	e := ret.Run(self.der)
	return ret, e
}
