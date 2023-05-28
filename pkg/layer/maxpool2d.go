package layer

import (
	"errors"

	"github.com/julioguillermo/neuralnetwork/pkg/activation"
	"github.com/julioguillermo/neuralnetwork/pkg/serialization"
	"github.com/julioguillermo/neuralnetwork/pkg/tensor"
)

type MaxPool2D struct {
	PreLayer Layer

	cInput  bool
	input   tensor.Tensor
	cOutput bool
	output  tensor.Tensor

	wSL bool
}

func NewMaxPool2D() *MaxPool2D {
	return &MaxPool2D{}
}

func (mp *MaxPool2D) GetOutShape() []int {
	shape := mp.PreLayer.GetOutShape()
	return []int{(shape[0]-1)/2 + 1, (shape[1]-1)/2 + 1, shape[2]}
}

func (mp *MaxPool2D) Build() error {
	return errors.New("this layer can not be used as model input")
}

func (mp *MaxPool2D) SetPrelayer(lay Layer) error {
	if mp.PreLayer != nil && lay != nil && !tensor.CompareShape(mp.PreLayer.GetOutShape(), lay.GetOutShape()) {
		return errors.New("invalid prelayers output shape")
	}
	mp.PreLayer = lay
	return nil
}

func (mp *MaxPool2D) Connect(p Layer) error {
	mp.PreLayer = p
	return nil
}

func (mp *MaxPool2D) GetActivation() activation.Activation {
	return activation.ActNull
}

func (mp *MaxPool2D) Reset() error {
	mp.cInput = false
	mp.cOutput = false
	return mp.PreLayer.Reset()
}

func (mp *MaxPool2D) FullReset() error {
	return mp.Reset()
}

func (mp *MaxPool2D) GetInput() tensor.Tensor {
	return mp.input
}

func (mp *MaxPool2D) getMax(i, x, y int) float64 {
	m, _ := mp.input.Get(x, y, i)
	var t float64
	if x+1 < mp.input.ShapeAt(0) {
		t, _ = mp.input.Get(x+1, y, i)
		if m < t {
			m = t
		}
	}
	if y+1 < mp.input.ShapeAt(1) {
		t, _ = mp.input.Get(x, y+1, i)
		if m < t {
			m = t
		}

		if x+1 < mp.input.ShapeAt(0) {
			t, _ = mp.input.Get(x+1, y+1, i)
			if m < t {
				m = t
			}
		}
	}

	return m
}

func (mp *MaxPool2D) Get(input tensor.Tensor) (tensor.Tensor, error) {
	if mp.cOutput {
		return mp.output, nil
	}
	var e error
	input, e = mp.PreLayer.Output(input)
	if e != nil {
		return nil, e
	}
	return mp.GetOne(input)
}

func (mp *MaxPool2D) GetOne(input tensor.Tensor) (tensor.Tensor, error) {
	if mp.cOutput {
		return mp.output, nil
	}
	mp.cOutput = true
	mp.input = input
	shape := mp.GetOutShape()
	mp.output = tensor.NewZeroTensor(shape...)
	for x := 0; x < shape[0]; x++ {
		for y := 0; y < shape[1]; y++ {
			for i := 0; i < shape[2]; i++ {
				mp.output.Set(mp.getMax(i, x*2, y*2), x, y, i)
			}
		}
	}
	return mp.output, nil
}

func (mp *MaxPool2D) Output(input tensor.Tensor) (tensor.Tensor, error) {
	if mp.cOutput {
		return mp.output, nil
	}
	return mp.Get(input)
}

func (mp *MaxPool2D) getDif(dif tensor.Tensor, x, y, i int) float64 {
	d := 0.0
	in, _ := mp.input.Get(x, y, i)
	out, _ := mp.output.Get(x/2, y/2, i)
	if in == out {
		d, _ = dif.Get(x/2, y/2, i)
	}
	return d
}

func (mp *MaxPool2D) SetDif(dif tensor.Tensor) {
	shape := mp.PreLayer.GetOutShape()
	pDif := tensor.NewZeroTensor(shape...)
	for i := 0; i < shape[2]; i++ {
		for x := 0; x < shape[0]; x++ {
			for y := 0; y < shape[1]; y++ {
				pDif.Set(mp.getDif(dif, x, y, i), x, y, i)
			}
		}
	}
	mp.PreLayer.SetDif(pDif)
}

func (mp *MaxPool2D) Dif() error {
	return mp.PreLayer.Dif()
}

func (mp *MaxPool2D) SetTrainable(bool) {}

func (mp *MaxPool2D) Fit(alpha, momentum float64) error {
	return mp.PreLayer.Fit(alpha, momentum)
}

func (mp *MaxPool2D) ResetSL() error {
	mp.wSL = false
	return mp.PreLayer.ResetSL()
}

func (mp *MaxPool2D) GetWeights() (serialization.Weights, error) {
	if mp.wSL {
		return serialization.Weights{}, nil
	}
	mp.wSL = true
	pw, e := mp.PreLayer.GetWeights()
	if e != nil {
		return serialization.Weights{}, e
	}
	return serialization.Weights{
		PreWeights: []serialization.Weights{pw},
	}, nil
}

func (mp *MaxPool2D) SetWeights(w serialization.Weights) error {
	if mp.wSL {
		return nil
	}
	mp.wSL = true
	if w.PreWeights != nil {
		if len(w.PreWeights) == 0 {
			return errors.New("ivalid preWeights len")
		}
		return mp.PreLayer.SetWeights(w.PreWeights[0])
	}
	return nil
}
