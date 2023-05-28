package model

import (
	"errors"
	"fmt"
	"math/rand"

	"github.com/julioguillermo/neuralnetwork/pkg/activation"
	"github.com/julioguillermo/neuralnetwork/pkg/layer"
	"github.com/julioguillermo/neuralnetwork/pkg/serialization"
	"github.com/julioguillermo/neuralnetwork/pkg/tensor"
)

type Sequential struct {
	PreLayer layer.Layer
	InLayer  layer.Layer
	OutLayer layer.Layer

	Trainable bool
}

func NewSequential() *Sequential {
	return &Sequential{
		PreLayer:  nil,
		InLayer:   nil,
		OutLayer:  nil,
		Trainable: true,
	}
}

func NewIOSequential(inLayer, outLayer layer.Layer) *Sequential {
	return &Sequential{
		PreLayer:  nil,
		InLayer:   inLayer,
		OutLayer:  outLayer,
		Trainable: true,
	}
}

// GetOutShape Sequential as Layer
func (sequential *Sequential) GetOutShape() []int {
	return sequential.OutLayer.GetOutShape()
}

func (sequential *Sequential) Build() error {
	sequential.PreLayer = nil
	return nil
}

func (sequential *Sequential) setSubPrelayer(lay layer.Layer) error {
	if sequential.PreLayer != nil && lay == nil || sequential.PreLayer == nil && lay != nil {
		return errors.New("invalid prelayers change")
	}
	if sequential.PreLayer != nil && lay != nil && !tensor.CompareShape(sequential.PreLayer.GetOutShape(), lay.GetOutShape()) {
		return errors.New("invalid prelayers output shape")
	}
	return sequential.InLayer.SetPrelayer(lay)
}

func (sequential *Sequential) SetPrelayer(lay layer.Layer) error {
	if sequential.PreLayer != nil && lay != nil && !tensor.CompareShape(sequential.PreLayer.GetOutShape(), lay.GetOutShape()) {
		return errors.New("invalid prelayers output shape")
	}
	return sequential.InLayer.SetPrelayer(lay)
}

func (sequential *Sequential) Connect(preLayer layer.Layer) error {
	sequential.PreLayer = preLayer
	if sequential.InLayer != nil {
		return sequential.InLayer.Connect(preLayer)
	}
	return nil
}

func (sequential *Sequential) GetActivation() activation.Activation {
	return sequential.OutLayer.GetActivation()
}

func (sequential *Sequential) Reset() error {
	return sequential.OutLayer.Reset()
}

func (sequential *Sequential) FullReset() error {
	e := sequential.setSubPrelayer(sequential.PreLayer)
	if e != nil {
		return e
	}
	return sequential.OutLayer.FullReset()
}

func (sequential *Sequential) GetInput() tensor.Tensor {
	return sequential.OutLayer.GetInput()
}

func (sequential *Sequential) Get(input tensor.Tensor) (tensor.Tensor, error) {
	return sequential.OutLayer.Get(input)
}

func (sequential *Sequential) GetOne(input tensor.Tensor) (tensor.Tensor, error) {
	return sequential.OutLayer.GetOne(input)
}

func (sequential *Sequential) Output(input tensor.Tensor) (tensor.Tensor, error) {
	return sequential.OutLayer.Output(input)
}

func (sequential *Sequential) SetDif(dif tensor.Tensor) {
	sequential.OutLayer.SetDif(dif)
}

func (sequential *Sequential) Dif() error {
	return sequential.OutLayer.Dif()
}

func (sequential *Sequential) SetTrainable(t bool) {
	sequential.OutLayer.SetTrainable(t)
	sequential.Trainable = t
}

func (sequential *Sequential) Fit(alpha float64, momentum float64) error {
	if sequential.Trainable {
		return sequential.OutLayer.Fit(alpha, momentum)
	} else if sequential.PreLayer != nil {
		return sequential.PreLayer.Fit(alpha, momentum)
	}
	return nil
}

func (sequential *Sequential) ResetSL() error {
	return sequential.OutLayer.ResetSL()
}

func (sequential *Sequential) GetWeights() (serialization.Weights, error) {
	return sequential.OutLayer.GetWeights()
}

func (sequential *Sequential) SetWeights(w serialization.Weights) error {
	return sequential.OutLayer.SetWeights(w)
}

// AddLayer Sequential as Model
func (sequential *Sequential) AddLayer(l layer.Layer) error {
	if sequential.InLayer == nil {
		sequential.InLayer = l
		sequential.OutLayer = l
		var err error
		if sequential.PreLayer == nil {
			err = l.Build()
		} else {
			err = l.Connect(sequential.PreLayer)
		}
		if err != nil {
			return err
		}
	} else {
		err := l.Connect(sequential.OutLayer)
		if err != nil {
			return err
		}
		sequential.OutLayer = l
	}
	return nil
}

func (sequential *Sequential) Predict(input tensor.Tensor) (tensor.Tensor, error) {
	e := sequential.setSubPrelayer(sequential.PreLayer)
	if e != nil {
		return nil, e
	}
	sequential.OutLayer.Reset()
	return sequential.OutLayer.Output(input)
}

func (sequential *Sequential) Train(inputs, targets []tensor.Tensor, alpha, momentum float64, epochs, batch, verbose int, loss func(outputs, targets tensor.Tensor) (tensor.Tensor, error), shuffle bool) (float64, error) {
	e := sequential.setSubPrelayer(sequential.PreLayer)
	if e != nil {
		return -1, e
	}
	if len(inputs) != len(targets) {
		return -1, errors.New("inputs and targets len are different")
	}

	tmp := make([]tensor.Tensor, len(inputs))
	copy(tmp, inputs)
	inputs = tmp
	tmp = make([]tensor.Tensor, len(targets))
	copy(tmp, targets)
	targets = tmp

	var bLoss tensor.Tensor
	var pLoss float64
	var err error
	if batch < 1 || len(inputs) < batch {
		batch = len(inputs)
	}
	for epoch := 1; epoch <= epochs; epoch++ {
		pLoss = 0
		if shuffle {
			rand.Shuffle(len(inputs), func(i, j int) {
				inputs[i], inputs[j] = inputs[j], inputs[i]
				targets[i], targets[j] = targets[j], targets[i]
			})
		}
		if verbose == 2 {
			fmt.Printf("\n\nEpoch: %d / %d [%.2f%%]\n", epoch, epochs, float64(epoch)/float64(epochs)*100.0)
		}
		for i := 0; i < batch; i++ {
			bLoss, err = sequential.TrainOne(inputs[i], targets[i], alpha, momentum, loss)
			if err != nil {
				return -1, err
			}
			if verbose == 2 {
				bLoss = bLoss.Abs()
				pLoss = bLoss.Sum() / float64(bLoss.Size())
				fmt.Printf("\rBatch: %d / %d [%.2f%%] => ( Loss: %f )", i, batch, float64(i+1)/float64(batch)*100.0, pLoss)
			} else {
				bLoss = bLoss.Abs()
				pLoss += bLoss.Sum() / float64(bLoss.Size())
			}
		}
		pLoss /= float64(batch)
		if verbose == 1 {
			fmt.Printf("\rEpoch: %d / %d [%.2f%%] => ( Loss: %s )", epoch, epochs, float64(epoch)/float64(epochs)*100.0, fmt.Sprint(pLoss))
		}
	}
	fmt.Println()
	return pLoss, nil
}

func (sequential *Sequential) TrainOne(input, target tensor.Tensor, alpha, momentum float64, loss func(outputs, targets tensor.Tensor) (tensor.Tensor, error)) (tensor.Tensor, error) {
	sequential.OutLayer.Reset()
	out, err := sequential.OutLayer.Output(input)
	if err != nil {
		return nil, err
	}
	out, err = loss(out, target)
	if err != nil {
		return nil, err
	}
	sequential.OutLayer.SetDif(out)
	err = sequential.OutLayer.Dif()
	if err != nil {
		return nil, err
	}
	err = sequential.OutLayer.Fit(alpha, momentum)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (sequential *Sequential) GetModelWeights() (serialization.Weights, error) {
	err := sequential.setSubPrelayer(sequential.PreLayer)
	if err != nil {
		return serialization.Weights{}, err
	}
	err = sequential.ResetSL()
	if err != nil {
		return serialization.Weights{}, err
	}
	return sequential.GetWeights()
}

func (sequential *Sequential) SetModelWeights(w serialization.Weights) error {
	err := sequential.setSubPrelayer(sequential.PreLayer)
	if err != nil {
		return err
	}
	err = sequential.ResetSL()
	if err != nil {
		return err
	}
	return sequential.SetWeights(w)
}
