package activation

import (
	"github.com/julioguillermo/neuralnetwork/pkg/tensor"
)

type LeakyRelu struct {
	Scale float64
}

func NewLeakyRelu(s float64) *LeakyRelu {
	return &LeakyRelu{s}
}

func (leaky *LeakyRelu) act(x float64, i int) (float64, error) {
	if x > 0 {
		return x, nil
	}
	return x * leaky.Scale, nil
}

func (leaky *LeakyRelu) der(x float64, i int) (float64, error) {
	if x > 0 {
		return 1, nil
	}
	return leaky.Scale, nil
}

func (leaky *LeakyRelu) Activate(input tensor.Tensor) (tensor.Tensor, error) {
	ret := input.Copy()
	e := ret.Run(leaky.act)
	return ret, e
}

func (leaky *LeakyRelu) Derive(input tensor.Tensor) (tensor.Tensor, error) {
	ret := input.Copy()
	e := ret.Run(leaky.der)
	return ret, e
}
