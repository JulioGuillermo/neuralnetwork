package main

import (
	"fmt"
	"runtime"

	"github.com/julioguillermo/neuralnetwork/pkg/activation"
	"github.com/julioguillermo/neuralnetwork/pkg/layer"
	"github.com/julioguillermo/neuralnetwork/pkg/loss"
	"github.com/julioguillermo/neuralnetwork/pkg/model"
	"github.com/julioguillermo/neuralnetwork/pkg/tensor"
)

func main() {
	inputs := []tensor.Tensor{
		tensor.NewTensor([]float64{
			0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
			0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
			1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		}, 10, 10, 1),
		tensor.NewTensor([]float64{
			1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
		}, 10, 10, 1),
	}

	runtime.GOMAXPROCS(runtime.NumCPU())
	m := model.NewSequential()
	/*m.AddLayer(layer.NewInDense(2, 50, activation.NewTanh()))
	m.AddLayer(layer.NewDense(50, activation.NewTanh()))
	//m.AddLayer(layer.NewDense(5, activation.NewLeakyRelu(0.1)))
	m.AddLayer(layer.NewDense(1, activation.NewTanh()))*/
	m.AddLayer(layer.NewInput(inputs[0].GetShape()...))
	m.AddLayer(layer.NewConv2D(10, 2, 2, 1, activation.NewTanh()))
	m.AddLayer(layer.NewMaxPool2D())
	m.AddLayer(layer.NewConv2D(10, 2, 2, 1, activation.NewTanh()))
	m.AddLayer(layer.NewMaxPool2D())
	//m.AddLayer(layer.NewConv2D(5, 2, 2, 1, activation.NewSigmoid()))
	m.AddLayer(layer.NewDense(1, activation.NewTanh()))

	targets := []tensor.Tensor{
		tensor.NewOneTensor(m.GetOutShape()...),
		tensor.NewZeroTensor(m.GetOutShape()...),
	}

	var e error

	fmt.Println("No trained:")
	for i, in := range inputs {
		is, _ := in.Str()
		ts, _ := targets[i].Str()
		o, e := m.Predict(in)
		if e != nil {
			fmt.Println(e.Error())
		} else {
			os, _ := o.Str()
			fmt.Printf("    %d: \033[31m%s\033[0m => (\033[32m%s\033[0m, \033[34m%s\033[0m)\n", i, is, ts, os)
		}
	}

	_, e = m.Train(inputs, targets, 0.01, 0.1, 10000, 0, 2, loss.L1, true)
	if e != nil {
		fmt.Println(e.Error())
	}
	/*w, e := m.GetModelWeights()
	if e != nil {
		fmt.Println(e.Error())
	}
	serialization.JsonSaveWeights(w, "./and_or_xor_nor.json")*/

	fmt.Println("Trained:")
	for i, in := range inputs {
		is, _ := in.Str()
		ts, _ := targets[i].Str()
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
