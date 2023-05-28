package activation

import (
	"math"

	"github.com/julioguillermo/neuralnetwork/pkg/tensor"
)

type Sigmoid struct{}

func NewSigmoid() *Sigmoid {
	return &Sigmoid{}
}

func (sigmoid *Sigmoid) act(x float64, i int) (float64, error) {
	return 1.0 / (1.0 + math.Exp(-x)), nil
}

func (sigmoid *Sigmoid) der(x float64, i int) (float64, error) {
	a, _ := sigmoid.act(x, i)
	return a * (1.0 - a), nil
}

func (sigmoid *Sigmoid) Activate(input tensor.Tensor) (tensor.Tensor, error) {
	ret := input.Copy()
	e := ret.Run(sigmoid.act)
	return ret, e
}

func (sigmoid *Sigmoid) Derive(input tensor.Tensor) (tensor.Tensor, error) {
	ret := input.Copy()
	e := ret.Run(sigmoid.der)
	return ret, e
}
