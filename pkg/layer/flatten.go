package layer

import (
	"errors"

	"github.com/julioguillermo/neuralnetwork/pkg/activation"
	"github.com/julioguillermo/neuralnetwork/pkg/serialization"
	"github.com/julioguillermo/neuralnetwork/pkg/tensor"
)

type Flatten struct {
	PreLayer Layer
	Shape    []int
	Size     int

	output  tensor.Tensor
	der     tensor.Tensor
	cOutput bool
	cDer    bool

	wSL bool
}

func NewFlatten() *Flatten {
	return &Flatten{}
}

func (flatten *Flatten) GetOutShape() []int {
	return flatten.Shape
}

func (flatten *Flatten) Build() error {
	return errors.New("this layer can not be used as model input")
}

func (flatten *Flatten) SetPrelayer(lay Layer) error {
	if flatten.PreLayer != nil && lay != nil && !tensor.CompareShape(flatten.PreLayer.GetOutShape(), lay.GetOutShape()) {
		return errors.New("invalid prelayers output shape")
	}
	flatten.PreLayer = lay
	return nil
}

func (flatten *Flatten) Connect(p Layer) error {
	flatten.PreLayer = p
	flatten.Shape = p.GetOutShape()
	flatten.Size = tensor.MulIndex(flatten.Shape, -1)
	return nil
}

func (flatten *Flatten) GetActivation() activation.Activation {
	return activation.ActNull
}

func (flatten *Flatten) Reset() error {
	flatten.cOutput = false
	flatten.cDer = false
	return flatten.PreLayer.Reset()
}

func (flatten *Flatten) FullReset() error {
	return flatten.Reset()
}

func (flatten *Flatten) GetInput() tensor.Tensor {
	return flatten.output
}

func (flatten *Flatten) Get(input tensor.Tensor) (tensor.Tensor, error) {
	if flatten.cOutput {
		return flatten.output, nil
	}
	var e error
	flatten.output, e = flatten.PreLayer.Output(input)
	if e != nil {
		return nil, e
	}
	flatten.output.Reshape(flatten.Size)
	flatten.cOutput = true
	return flatten.output, nil
}

func (flatten *Flatten) GetOne(input tensor.Tensor) (tensor.Tensor, error) {
	if flatten.cDer {
		return flatten.der, nil
	}
	var e error
	flatten.der, e = flatten.PreLayer.GetOne(flatten.PreLayer.GetInput())
	if e != nil {
		return nil, e
	}
	flatten.der, e = flatten.PreLayer.GetActivation().Derive(flatten.der)
	if e != nil {
		return nil, e
	}
	flatten.der.Reshape(flatten.Size)
	flatten.cDer = true
	return flatten.der, nil
}

func (flatten *Flatten) Output(input tensor.Tensor) (tensor.Tensor, error) {
	return flatten.Get(input)
}

func (flatten *Flatten) SetDif(dif tensor.Tensor) {
	dif.Reshape(flatten.Shape...)
	flatten.PreLayer.SetDif(dif)
}

func (flatten *Flatten) Dif() error {
	flatten.PreLayer.Dif()
	return nil
}

func (flatten *Flatten) SetTrainable(bool) {}

func (flatten *Flatten) Fit(alpha, momentum float64) error {
	return flatten.PreLayer.Fit(alpha, momentum)
}

func (flatten *Flatten) ResetSL() error {
	flatten.wSL = false
	return flatten.PreLayer.ResetSL()
}

func (flatten *Flatten) GetWeights() (serialization.Weights, error) {
	if flatten.wSL {
		return serialization.Weights{}, nil
	}
	flatten.wSL = true

	pw, e := flatten.PreLayer.GetWeights()
	if e != nil {
		return serialization.Weights{}, e
	}

	return serialization.Weights{
		PreWeights: []serialization.Weights{pw},
	}, nil
}

func (flatten *Flatten) SetWeights(w serialization.Weights) error {
	if flatten.wSL {
		return nil
	}
	flatten.wSL = true
	if w.PreWeights != nil {
		if len(w.PreWeights) == 0 {
			return errors.New("ivalid preWeights len")
		}
		return flatten.PreLayer.SetWeights(w.PreWeights[0])
	}
	return nil
}
