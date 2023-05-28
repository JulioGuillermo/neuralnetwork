package layer

import (
	"errors"

	"github.com/julioguillermo/neuralnetwork/pkg/activation"
	"github.com/julioguillermo/neuralnetwork/pkg/serialization"
	"github.com/julioguillermo/neuralnetwork/pkg/tensor"
)

type Reshape struct {
	PreLayer Layer
	InShape  []int
	Shape    []int

	cInput  bool
	input   tensor.Tensor
	cOutput bool
	output  tensor.Tensor
	cDer    bool
	der     tensor.Tensor

	wSL bool
}

func NewReshape(shape ...int) *Reshape {
	return &Reshape{
		Shape: shape,
	}
}

func (reshape *Reshape) GetOutShape() []int {
	return reshape.Shape
}

func (reshape *Reshape) Build() error {
	return errors.New("this layer can not be used as model input")
}

func (reshape *Reshape) SetPrelayer(lay Layer) error {
	if reshape.PreLayer != nil && lay != nil && !tensor.CompareShape(reshape.PreLayer.GetOutShape(), lay.GetOutShape()) {
		return errors.New("invalid prelayers output shape")
	}
	reshape.PreLayer = lay
	return nil
}

func (reshape *Reshape) Connect(p Layer) error {
	reshape.PreLayer = p
	reshape.InShape = p.GetOutShape()
	return nil
}

func (reshape *Reshape) GetActivation() activation.Activation {
	return activation.ActNull
}

func (reshape *Reshape) Reset() error {
	reshape.cInput = false
	reshape.cOutput = false
	reshape.cDer = false
	return reshape.PreLayer.Reset()
}

func (reshape *Reshape) FullReset() error {
	return reshape.Reset()
}

func (reshape *Reshape) GetInput() tensor.Tensor {
	return reshape.input
}

func (reshape *Reshape) Get(input tensor.Tensor) (tensor.Tensor, error) {
	if reshape.cOutput {
		return reshape.output, nil
	}
	var e error
	reshape.output, e = reshape.PreLayer.Output(input)
	if e != nil {
		return nil, e
	}
	reshape.input = reshape.output.Copy()
	reshape.output.Reshape(reshape.Shape...)
	reshape.cOutput = true
	return reshape.output, nil
}

func (reshape *Reshape) GetOne(input tensor.Tensor) (tensor.Tensor, error) {
	if reshape.cDer {
		return reshape.der, nil
	}
	var e error
	reshape.der, e = reshape.PreLayer.GetOne(reshape.PreLayer.GetInput())
	if e != nil {
		return nil, e
	}
	reshape.der, e = reshape.PreLayer.GetActivation().Derive(reshape.der)
	if e != nil {
		return nil, e
	}
	reshape.der.Reshape(reshape.Shape...)
	reshape.cDer = true
	return reshape.der, nil
}

func (reshape *Reshape) Output(input tensor.Tensor) (tensor.Tensor, error) {
	return reshape.Get(input)
}

func (reshape *Reshape) SetDif(dif tensor.Tensor) {
	dif.Reshape(reshape.InShape...)
	reshape.PreLayer.SetDif(dif)
}

func (reshape *Reshape) Dif() error {
	return reshape.PreLayer.Dif()
}

func (reshape *Reshape) SetTrainable(bool) {}

func (reshape *Reshape) Fit(alpha, momentum float64) error {
	return reshape.PreLayer.Fit(alpha, momentum)
}

func (reshape *Reshape) ResetSL() error {
	reshape.wSL = false
	return reshape.PreLayer.ResetSL()
}

func (reshape *Reshape) GetWeights() (serialization.Weights, error) {
	if reshape.wSL {
		return serialization.Weights{}, nil
	}
	reshape.wSL = true

	pw, e := reshape.PreLayer.GetWeights()
	if e != nil {
		return serialization.Weights{}, e
	}

	return serialization.Weights{
		PreWeights: []serialization.Weights{pw},
	}, nil
}

func (reshape *Reshape) SetWeights(w serialization.Weights) error {
	if reshape.wSL {
		return nil
	}
	reshape.wSL = true
	if w.PreWeights != nil {
		if len(w.PreWeights) == 0 {
			return errors.New("ivalid preWeights len")
		}
		return reshape.PreLayer.SetWeights(w.PreWeights[0])
	}
	return nil
}
