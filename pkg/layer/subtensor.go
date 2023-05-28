package layer

import (
	"errors"

	"github.com/julioguillermo/neuralnetwork/pkg/activation"
	"github.com/julioguillermo/neuralnetwork/pkg/serialization"
	"github.com/julioguillermo/neuralnetwork/pkg/tensor"
)

type SubTensor struct {
	PreLayer Layer
	InShape  []int
	Shape    []int
	InLen    int
	Offset   int
	Index    int

	input  tensor.Tensor
	output tensor.Tensor
	der    tensor.Tensor
	cIn    bool
	cOut   bool
	cDer   bool

	wSL bool
}

func NewSubTensor(index int) *SubTensor {
	return &SubTensor{
		Index: index,
	}
}

func (sub *SubTensor) GetOutShape() []int {
	return sub.Shape
}

func (sub *SubTensor) Build() error {
	return errors.New("this layer can not be used as input")
}

func (sub *SubTensor) SetPrelayer(lay Layer) error {
	if sub.PreLayer != nil && lay != nil && !tensor.CompareShape(sub.PreLayer.GetOutShape(), lay.GetOutShape()) {
		return errors.New("invalid prelayers output shape")
	}
	sub.PreLayer = lay
	return nil
}

func (sub *SubTensor) Connect(preLayer Layer) error {
	sub.PreLayer = preLayer
	sub.InShape = preLayer.GetOutShape()
	sub.Shape = sub.InShape[1:]
	sub.InLen = tensor.MulIndex(sub.InShape, -1)
	sub.Offset = sub.Index * sub.InShape[0]
	return nil
}

func (sub *SubTensor) GetActivation() activation.Activation {
	return activation.ActNull
}

func (sub *SubTensor) Reset() error {
	sub.cIn = false
	sub.cOut = false
	sub.cDer = false
	return sub.PreLayer.Reset()
}

func (sub *SubTensor) FullReset() error {
	return sub.Reset()
}

func (sub *SubTensor) GetInput() tensor.Tensor {
	return sub.input
}

func (sub *SubTensor) Get(input tensor.Tensor) (tensor.Tensor, error) {
	if sub.cOut {
		return sub.output, nil
	}
	var e error
	sub.input, e = sub.PreLayer.Output(input)
	if e != nil {
		return nil, e
	}
	sub.input.Reshape(sub.InShape...)
	sub.output, e = sub.input.GetSubTensor(sub.Index)
	if e != nil {
		return nil, e
	}
	sub.cIn = true
	sub.cOut = true
	return sub.output, nil
}

func (sub *SubTensor) GetOne(input tensor.Tensor) (tensor.Tensor, error) {
	if sub.cDer {
		return sub.der, nil
	}
	var e error
	sub.der, e = sub.PreLayer.GetOne(sub.PreLayer.GetInput())
	if e != nil {
		return nil, e
	}
	sub.der.Reshape(sub.InShape...)
	sub.der, e = sub.der.GetSubTensor(sub.Index)
	if e != nil {
		return nil, e
	}
	sub.der, e = sub.PreLayer.GetActivation().Derive(sub.der)
	if e != nil {
		return nil, e
	}
	sub.cDer = true
	return sub.der, nil
}

func (sub *SubTensor) Output(input tensor.Tensor) (tensor.Tensor, error) {
	return sub.Get(input)
}

func (sub *SubTensor) SetDif(dif tensor.Tensor) {
	data := make([]float64, sub.InLen)
	ddat := dif.GetData()
	for i, d := range ddat {
		data[sub.Offset+i] = d
	}
	dif = tensor.NewTensor(data, sub.InShape...)
	sub.PreLayer.SetDif(dif)
}

func (sub *SubTensor) Dif() error {
	return sub.PreLayer.Dif()
}

func (sub *SubTensor) SetTrainable(bool) {}

func (sub *SubTensor) Fit(alpha float64, momentum float64) error {
	return sub.PreLayer.Fit(alpha, momentum)
}

func (sub *SubTensor) ResetSL() error {
	sub.wSL = false
	return sub.PreLayer.ResetSL()
}

func (sub *SubTensor) GetWeights() (serialization.Weights, error) {
	if sub.wSL {
		return serialization.Weights{}, nil
	}
	sub.wSL = true

	pw, e := sub.PreLayer.GetWeights()
	if e != nil {
		return serialization.Weights{}, e
	}

	return serialization.Weights{
		PreWeights: []serialization.Weights{pw},
	}, nil
}

func (sub *SubTensor) SetWeights(w serialization.Weights) error {
	if sub.wSL {
		return nil
	}
	sub.wSL = true
	if w.PreWeights != nil {
		if len(w.PreWeights) == 0 {
			return errors.New("ivalid preWeights len")
		}
		return sub.PreLayer.SetWeights(w.PreWeights[0])
	}
	return nil
}
