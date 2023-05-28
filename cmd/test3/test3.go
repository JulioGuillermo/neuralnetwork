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
		tensor.NewOneTensor(1, 1, 1),
		tensor.NewZeroTensor(1, 1, 1),
	}
	y := []tensor.Tensor{
		tensor.NewTensor([]float64{
			1, 0, 0, 0, 0,
			0, 1, 0, 0, 0,
			0, 0, 1, 0, 0,
			0, 0, 0, 1, 0,
			0, 0, 0, 0, 1,
		}, 5, 5, 1),
		tensor.NewTensor([]float64{
			0, 0, 0, 0, 1,
			0, 0, 0, 1, 0,
			0, 0, 1, 0, 0,
			0, 1, 0, 0, 0,
			1, 0, 0, 0, 0,
		}, 5, 5, 1),
	}

	m := model.NewSequential()
	m.AddLayer(layer.NewInput(x[0].GetShape()...))
	m.AddLayer(layer.NewDeconv2D(100, 3, 3, 1, activation.NewTanh()))
	m.AddLayer(layer.NewDeconv2D(1, 3, 3, 1, activation.NewTanh()))
	/*m.AddLayer(layer.NewDeconv2D(30, 3, 3, 1, activation.NewTanh()))
	m.AddLayer(layer.NewDeconv2D(30, 2, 2, 1, activation.NewSigmoid()))
	m.AddLayer(layer.NewDeconv2D(1, 2, 2, 1, activation.NewTanh()))
	*/
	_, e := m.Train(x, y, 0.001, 0.5, 10000, 0, 1, loss.L1, false)
	if e != nil {
		fmt.Println(e)
	}

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
