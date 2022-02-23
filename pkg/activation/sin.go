package activation

import (
	"math"

	"github.com/JulioGuillermo/neuralnetwork/pkg/tensor"
)

type Sin struct{}

func NewSin() *Sin {
	return &Sin{}
}

func (self *Sin) act(x float64, i int) (float64, error) {
	return math.Sin(x), nil
}

func (self *Sin) der(x float64, i int) (float64, error) {
	return math.Cos(x), nil
}

func (self *Sin) Activate(input tensor.Tensor) (tensor.Tensor, error) {
	ret := input.Copy()
	e := ret.Run(self.act)
	return ret, e
}

func (self *Sin) Derive(input tensor.Tensor) (tensor.Tensor, error) {
	ret := input.Copy()
	e := ret.Run(self.der)
	return ret, e
}
