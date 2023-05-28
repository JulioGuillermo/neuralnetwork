package main

import (
	"fmt"

	"github.com/julioguillermo/neuralnetwork/pkg/activation"
	"github.com/julioguillermo/neuralnetwork/pkg/layer"
	"github.com/julioguillermo/neuralnetwork/pkg/loss"
	"github.com/julioguillermo/neuralnetwork/pkg/model"
	"github.com/julioguillermo/neuralnetwork/pkg/serialization"
	"github.com/julioguillermo/neuralnetwork/pkg/tensor"
)

func getDS() ([]tensor.Tensor, []tensor.Tensor) {
	fmt.Println("Creating DS...")
	x := []tensor.Tensor{
		tensor.NewTensor([]float64{0, 0}, 2),
		tensor.NewTensor([]float64{0, 1}, 2),
		tensor.NewTensor([]float64{1, 0}, 2),
		tensor.NewTensor([]float64{1, 1}, 2),
	}
	y := []tensor.Tensor{
		tensor.NewTensor([]float64{0}, 1),
		tensor.NewTensor([]float64{1}, 1),
		tensor.NewTensor([]float64{1}, 1),
		tensor.NewTensor([]float64{0}, 1),
	}
	return x, y
}

func getModel() model.Model {
	fmt.Println("Creating Model...")
	m := model.NewSequential()
	m.AddLayer(layer.NewInDense(2, 3, activation.NewSigmoid()))
	m.AddLayer(layer.NewDense(5, activation.NewSigmoid()))
	m.AddLayer(layer.NewDense(1, activation.NewSigmoid()))
	return m
}

func train(m model.Model, x, y []tensor.Tensor) {
	fmt.Println("Training...")
	m.Train(x, y, 0.01, 0.5, 100000, 0, 1, loss.L1, false)
	w, e := m.GetModelWeights()
	if e != nil {
		fmt.Println(e)
	}
	e = serialization.BinSaveWeights(w, "model.gob")
	if e != nil {
		fmt.Println(e)
	}
}

func load(m model.Model) error {
	fmt.Println("Loading...")
	w, e := serialization.BinLoadWeights("model.gob")
	if e != nil {
		return e
	}
	e = m.SetModelWeights(w)
	if e != nil {
		return e
	}
	return nil
}

func test(m model.Model, x, y []tensor.Tensor) {
	fmt.Println("Testing...")
	for i, in := range x {
		o, _ := m.Predict(in)
		os, _ := o.Str()
		xs, _ := in.Str()
		ys, _ := y[i].Str()
		fmt.Printf("\033[34m%s\033[0m : \033[33m%s\033[0m => \033[32m%s\033[0m\n", xs, os, ys)
	}
}

func main() {
	x, y := getDS()
	m := getModel()
	test(m, x, y)

	train(m, x, y)
	/*e := load(m)
	  if e != nil {
	      train(m, x, y)
	  }*/

	test(m, x, y)
}
