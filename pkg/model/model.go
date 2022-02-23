package model

import (
	"github.com/JulioGuillermo/neuralnetwork/pkg/layer"
	"github.com/JulioGuillermo/neuralnetwork/pkg/serialization"
	"github.com/JulioGuillermo/neuralnetwork/pkg/tensor"
)

type Model interface {
	AddLayer(layer.Layer) error
	Predict(tensor.Tensor) (tensor.Tensor, error)
	Train(inputs, targets []tensor.Tensor, alpha, momentum float64, epochs, batch, verbose int, loss func(outputs, targets tensor.Tensor) (tensor.Tensor, error)) error

	GetModelWeights() (serialization.Weights, error)
	SetModelWeights(w serialization.Weights) error
}
