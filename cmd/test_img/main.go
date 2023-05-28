package main

import (
	"github.com/julioguillermo/neuralnetwork/pkg/activation"
	"github.com/julioguillermo/neuralnetwork/pkg/layer"
	"github.com/julioguillermo/neuralnetwork/pkg/model"
	"github.com/julioguillermo/neuralnetwork/pkg/tensor"
)

func main() {
	m := model.NewSequential()
	m.AddLayer(layer.NewInput(5, 5, 5))
	m.AddLayer(layer.NewDeconv2D(10, 3, 3, 1, activation.NewTanh()))
	m.AddLayer(layer.NewDeconv2D(10, 5, 5, 2, activation.NewTanh()))
	m.AddLayer(layer.NewDeconv2D(10, 5, 5, 2, activation.NewTanh()))
	m.AddLayer(layer.NewDeconv2D(3, 5, 5, 2, activation.NewTanh()))

	in := tensor.NewRandTensor(0, 1, 5, 5, 5)
	out, _ := m.Predict(in)

	tensor.SaveTensorAsJPEG(out, "img")
}
