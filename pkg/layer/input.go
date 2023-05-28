package layer

import (
	"errors"

	"github.com/julioguillermo/neuralnetwork/pkg/activation"
	"github.com/julioguillermo/neuralnetwork/pkg/serialization"
	"github.com/julioguillermo/neuralnetwork/pkg/tensor"
)

type Input struct {
	Shape []int
	Size  int
	input tensor.Tensor
}

func NewInput(shape ...int) *Input {
	return &Input{
		Shape: shape,
		Size:  tensor.MulIndex(shape, -1),
	}
}

func (inlay *Input) GetOutShape() []int {
	return inlay.Shape
}

func (inlay *Input) Build() error {
	return nil
}

func (inlay *Input) SetPrelayer(lay Layer) error {
	if lay == nil {
		return nil
	}
	return errors.New("invalid prelayer change")
}

func (inlay *Input) Connect(preLayer Layer) error {
	return errors.New("this layer can not be used as hidden layer")
}

func (inlay *Input) GetActivation() activation.Activation {
	return activation.ActNull
}

func (inlay *Input) Reset() error {
	return nil
}

func (inlay *Input) FullReset() error {
	return inlay.Reset()
}

func (inlay *Input) GetInput() tensor.Tensor {
	return inlay.input
}

func (inlay *Input) Get(input tensor.Tensor) (tensor.Tensor, error) {
	if inlay.Size != input.Size() {
		return nil, errors.New("incompatible input shape")
	}
	inlay.input = input.Copy()
	return inlay.input, nil
}

func (inlay *Input) GetOne(input tensor.Tensor) (tensor.Tensor, error) {
	return inlay.Get(input)
}

func (inlay *Input) Output(input tensor.Tensor) (tensor.Tensor, error) {
	return inlay.Get(input)
}

func (inlay *Input) SetDif(dif tensor.Tensor) {

}

func (inlay *Input) Dif() error {
	return nil
}

func (inlay *Input) SetTrainable(bool) {

}

func (inlay *Input) Fit(alpha float64, momentum float64) error {
	return nil
}

func (inlay *Input) ResetSL() error {
	return nil
}

func (inlay *Input) GetWeights() (serialization.Weights, error) {
	return serialization.Weights{}, nil
}

func (inlay *Input) SetWeights(w serialization.Weights) error {
	return nil
}
