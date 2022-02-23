package activation

import (
	"github.com/JulioGuillermo/neuralnetwork/pkg/tensor"
)

type LeakyRelu struct {
	Scale float64
}

func NewLeakyRelu(s float64) *LeakyRelu {
	return &LeakyRelu{s}
}

func (self *LeakyRelu) act(x float64, i int) (float64, error) {
	if x > 0 {
		return x, nil
	}
	return x * self.Scale, nil
}

func (self *LeakyRelu) der(x float64, i int) (float64, error) {
	if x > 0 {
		return 1, nil
	}
	return self.Scale, nil
}

func (self *LeakyRelu) Activate(input tensor.Tensor) (tensor.Tensor, error) {
	ret := input.Copy()
	e := ret.Run(self.act)
	return ret, e
}

func (self *LeakyRelu) Derive(input tensor.Tensor) (tensor.Tensor, error) {
	ret := input.Copy()
	e := ret.Run(self.der)
	return ret, e
}
