package model

import (
	"github.com/julioguillermo/neuralnetwork/pkg/layer"
	"github.com/julioguillermo/neuralnetwork/pkg/serialization"
	"github.com/julioguillermo/neuralnetwork/pkg/tensor"
)

type Model interface {
	AddLayer(layer.Layer) error
	Predict(tensor.Tensor) (tensor.Tensor, error)
	Train(inputs, targets []tensor.Tensor, alpha, momentum float64, epochs, batch, verbose int, loss func(outputs, targets tensor.Tensor) (tensor.Tensor, error), shuffle bool) (float64, error)
	TrainOne(input, target tensor.Tensor, alpha, momentum float64, loss func(outputs, targets tensor.Tensor) (tensor.Tensor, error)) (tensor.Tensor, error)
	FullReset() error

	SetTrainable(bool)

	GetModelWeights() (serialization.Weights, error)
	SetModelWeights(w serialization.Weights) error
}
