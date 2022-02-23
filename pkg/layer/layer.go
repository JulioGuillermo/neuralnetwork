package layer

import (
	"github.com/JulioGuillermo/neuralnetwork/pkg/activation"
	"github.com/JulioGuillermo/neuralnetwork/pkg/serialization"
	"github.com/JulioGuillermo/neuralnetwork/pkg/tensor"
)

type Layer interface {
	GetOutShape() []int

	Build() error
	Connect(preLayer Layer) error

	GetActivation() activation.Activation

	Reset() error

	GetInput() tensor.Tensor

	Get(input tensor.Tensor) (tensor.Tensor, error)
	GetOne(input tensor.Tensor) (tensor.Tensor, error)
	Output(input tensor.Tensor) (tensor.Tensor, error)

	SetDif(dif tensor.Tensor)
	Dif() error

	SetTrainable(bool)
	Fit(alpha float64, momentum float64) error

	ResetSL() error
	GetWeights() (serialization.Weights, error)
	SetWeights(serialization.Weights) error
}
