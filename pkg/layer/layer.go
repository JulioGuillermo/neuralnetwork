package layer

import (
	"github.com/julioguillermo/neuralnetwork/pkg/activation"
	"github.com/julioguillermo/neuralnetwork/pkg/serialization"
	"github.com/julioguillermo/neuralnetwork/pkg/tensor"
)

type Layer interface {
	GetOutShape() []int

	Build() error
	SetPrelayer(Layer) error
	Connect(Layer) error

	GetActivation() activation.Activation

	Reset() error
	FullReset() error

	GetInput() tensor.Tensor

	Get(tensor.Tensor) (tensor.Tensor, error)
	GetOne(tensor.Tensor) (tensor.Tensor, error)
	Output(tensor.Tensor) (tensor.Tensor, error)

	SetDif(tensor.Tensor)
	Dif() error

	SetTrainable(bool)
	Fit(float64, float64) error

	ResetSL() error
	GetWeights() (serialization.Weights, error)
	SetWeights(serialization.Weights) error
}
