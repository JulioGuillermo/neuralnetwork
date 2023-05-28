package activation

import (
	"math"

	"github.com/julioguillermo/neuralnetwork/pkg/tensor"
)

type Sin struct{}

func NewSin() *Sin {
	return &Sin{}
}

func (sin *Sin) act(x float64, i int) (float64, error) {
	return math.Sin(x), nil
}

func (sin *Sin) der(x float64, i int) (float64, error) {
	return math.Cos(x), nil
}

func (sin *Sin) Activate(input tensor.Tensor) (tensor.Tensor, error) {
	ret := input.Copy()
	e := ret.Run(sin.act)
	return ret, e
}

func (sin *Sin) Derive(input tensor.Tensor) (tensor.Tensor, error) {
	ret := input.Copy()
	e := ret.Run(sin.der)
	return ret, e
}
