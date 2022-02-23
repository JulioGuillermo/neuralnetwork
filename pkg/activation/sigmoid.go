package activation

import (
	"math"

	"github.com/JulioGuillermo/neuralnetwork/pkg/tensor"
)

type Sigmoid struct{}

func NewSigmoid() *Sigmoid {
	return &Sigmoid{}
}

func (self *Sigmoid) act(x float64, i int) (float64, error) {
	return 1.0 / (1.0 + math.Exp(-x)), nil
}

func (self *Sigmoid) der(x float64, i int) (float64, error) {
	a, _ := self.act(x, i)
	return a * (1.0 - a), nil
}

func (self *Sigmoid) Activate(input tensor.Tensor) (tensor.Tensor, error) {
	ret := input.Copy()
	e := ret.Run(self.act)
	return ret, e
}

func (self *Sigmoid) Derive(input tensor.Tensor) (tensor.Tensor, error) {
	ret := input.Copy()
	e := ret.Run(self.der)
	return ret, e
}
