package activation

import (
	"math"

	"github.com/julioguillermo/neuralnetwork/pkg/tensor"
)

type Tanh struct{}

func NewTanh() *Tanh {
	return &Tanh{}
}

func (tanh *Tanh) act(x float64, i int) (float64, error) {
	return math.Tanh(x), nil
}

func (tanh *Tanh) der(x float64, i int) (float64, error) {
	return 1.0 / math.Pow(math.Cosh(x), 2.0), nil
}

func (tanh *Tanh) Activate(input tensor.Tensor) (tensor.Tensor, error) {
	ret := input.Copy()
	e := ret.Run(tanh.act)
	return ret, e
}

func (tanh *Tanh) Derive(input tensor.Tensor) (tensor.Tensor, error) {
	ret := input.Copy()
	e := ret.Run(tanh.der)
	return ret, e
}
