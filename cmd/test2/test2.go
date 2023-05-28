package main

import (
	"fmt"

	"github.com/julioguillermo/neuralnetwork/pkg/activation"
	"github.com/julioguillermo/neuralnetwork/pkg/layer"
	"github.com/julioguillermo/neuralnetwork/pkg/loss"
	"github.com/julioguillermo/neuralnetwork/pkg/model"
	"github.com/julioguillermo/neuralnetwork/pkg/tensor"
)

func main() {
	x := []tensor.Tensor{
		tensor.NewTensor([]float64{1, 0, 0, 0, 0}, 5),
		tensor.NewTensor([]float64{0, 1, 0, 0, 0}, 5),
		tensor.NewTensor([]float64{0, 0, 1, 0, 0}, 5),
		tensor.NewTensor([]float64{0, 0, 0, 1, 0}, 5),
		tensor.NewTensor([]float64{0, 0, 0, 0, 1}, 5),
		tensor.NewTensor([]float64{0, 0, 0, 1, 0}, 5),
		tensor.NewTensor([]float64{0, 0, 1, 0, 0}, 5),
		tensor.NewTensor([]float64{0, 1, 0, 0, 0}, 5),
		tensor.NewTensor([]float64{1, 0, 0, 0, 0}, 5),
	}
	y := []tensor.Tensor{
		tensor.NewTensor([]float64{0, 1, 0, 0, 0}, 5),
		tensor.NewTensor([]float64{0, 0, 1, 0, 0}, 5),
		tensor.NewTensor([]float64{0, 0, 0, 1, 0}, 5),
		tensor.NewTensor([]float64{0, 0, 0, 0, 1}, 5),
		tensor.NewTensor([]float64{0, 0, 0, 1, 0}, 5),
		tensor.NewTensor([]float64{0, 0, 1, 0, 0}, 5),
		tensor.NewTensor([]float64{0, 1, 0, 0, 0}, 5),
		tensor.NewTensor([]float64{1, 0, 0, 0, 0}, 5),
		tensor.NewTensor([]float64{0, 1, 0, 0, 0}, 5),
	}
	m := model.NewSequential()
	m.AddLayer(layer.NewInput(5))
	m.AddLayer(layer.NewRecurrent2(10, activation.NewTanh()))
	//m.AddLayer(layer.NewInRecurrent(5, 10, activation.NewTanh()))
	m.AddLayer(layer.NewRecurrent2(5, activation.NewTanh()))

	m.Train(x, y, 0.01, 0.5, 10000, 0, 1, loss.L1, false)

	for i, in := range x {
		is, _ := in.Str()
		ts, _ := y[i].Str()
		o, e := m.Predict(in)
		if e != nil {
			fmt.Println(e.Error())
		} else {
			os, _ := o.Str()
			//fmt.Printf("    %d: %s => (%s, %s)\n", i, is, ts, os)
			fmt.Printf("    %d: \033[31m%s\033[0m => (\033[32m%s\033[0m, \033[34m%s\033[0m)\n", i, is, ts, os)
		}
	}
}
