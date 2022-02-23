package activation

import (
	"math"

	"github.com/JulioGuillermo/neuralnetwork/pkg/tensor"
)

type Tanh struct{}

func NewTanh() *Tanh {
	return &Tanh{}
}

func (self *Tanh) act(x float64, i int) (float64, error) {
	return math.Tanh(x), nil
}

func (self *Tanh) der(x float64, i int) (float64, error) {
	return 1.0 / math.Pow(math.Cosh(x), 2.0), nil
}

func (self *Tanh) Activate(input tensor.Tensor) (tensor.Tensor, error) {
	ret := input.Copy()
	e := ret.Run(self.act)
	return ret, e
}

func (self *Tanh) Derive(input tensor.Tensor) (tensor.Tensor, error) {
	ret := input.Copy()
	e := ret.Run(self.der)
	return ret, e
}
