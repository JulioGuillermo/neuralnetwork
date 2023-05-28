package layer

import (
	"errors"

	"github.com/julioguillermo/neuralnetwork/pkg/activation"
	"github.com/julioguillermo/neuralnetwork/pkg/serialization"
	"github.com/julioguillermo/neuralnetwork/pkg/tensor"
)

type Concat struct {
	PreLayers []Layer
	Shape     []int
	OShape    []int

	output  tensor.Tensor
	cOutput bool
	der     tensor.Tensor
	cDer    bool
	dif     tensor.Tensor
	cDif    int

	wSL bool
}

func NewConcat(layers ...Layer) (*Concat, error) {
	if len(layers) < 1 {
		return nil, errors.New("no layers given")
	}
	shape := layers[0].GetOutShape()
	for _, l := range layers {
		tmp := l.GetOutShape()
		if len(shape) != len(tmp) {
			return nil, errors.New("incompatible layers outputs shape dimensions")
		}
		for i, s := range shape {
			if s != tmp[i] {
				return nil, errors.New("incompatible layers outputs shape")
			}
		}
	}
	shape = append([]int{1}, shape...)

	oshape := make([]int, len(shape))
	copy(oshape, shape)
	oshape[0] *= len(layers)

	return &Concat{
		PreLayers: layers,
		Shape:     shape,
		OShape:    oshape,
	}, nil
}

func (concat *Concat) GetOutShape() []int {
	return concat.OShape
}

func (concat *Concat) Build() error {
	return errors.New("this layer can not be used as model input")
}

func (concat *Concat) SetPrelayer(lay Layer) error {
	if lay == nil {
		return nil
	}
	return errors.New("invalid prelayer change")
}

func (concat *Concat) Connect(p Layer) error {
	return nil
}

func (concat *Concat) GetActivation() activation.Activation {
	return activation.ActNull
}

func (concat *Concat) Reset() error {
	if concat.cOutput || concat.cDif != 0 {
		concat.cOutput = false
		concat.cDif = 0
		var e error
		for _, l := range concat.PreLayers {
			e = l.Reset()
			if e != nil {
				return e
			}
		}
	}
	return nil
}

func (concat *Concat) FullReset() error {
	return concat.Reset()
}

func (concat *Concat) GetInput() tensor.Tensor {
	return concat.output
}

func (concat *Concat) Get(input tensor.Tensor) (tensor.Tensor, error) {
	if concat.cOutput {
		return concat.output, nil
	}
	outs := make([]tensor.Tensor, len(concat.PreLayers))
	var e error
	for i, l := range concat.PreLayers {
		outs[i], e = l.Output(input)
		if e != nil {
			return nil, e
		}
		outs[i].Reshape(concat.Shape...)
	}
	concat.output, e = tensor.ConcatTensors(0, outs...)
	concat.cOutput = true
	return concat.output, e
}

func (concat *Concat) GetOne(input tensor.Tensor) (tensor.Tensor, error) {
	if concat.cDer {
		return concat.der, nil
	}
	outs := make([]tensor.Tensor, len(concat.PreLayers))
	var e error
	for i, l := range concat.PreLayers {
		outs[i], e = l.GetOne(l.GetInput())
		if e != nil {
			return nil, e
		}
		outs[i].Reshape(concat.Shape...)
		outs[i], e = l.GetActivation().Derive(outs[i])
		if e != nil {
			return nil, e
		}
	}
	concat.der, e = tensor.ConcatTensors(0, outs...)
	concat.cDer = true
	return concat.der, e
}

func (concat *Concat) Output(input tensor.Tensor) (tensor.Tensor, error) {
	return concat.Get(input)
}

func (concat *Concat) SetDif(dif tensor.Tensor) {
	dif.Reshape(concat.OShape...)
	if concat.cDif == 0 {
		concat.dif = dif
	} else {
		concat.dif.AddTensor(dif)
	}
	concat.cDif++
}

func (concat *Concat) Dif() error {
	var t tensor.Tensor
	var e error
	for i, l := range concat.PreLayers {
		t, e = concat.dif.GetSubTensor(i)
		if e != nil {
			return e
		}
		l.SetDif(t)
		e = l.Dif()
		if e != nil {
			return e
		}
	}
	return nil
}

func (concat *Concat) SetTrainable(bool) {}

func (concat *Concat) Fit(alpha, momentum float64) error {
	var e error
	for _, l := range concat.PreLayers {
		e = l.Fit(alpha, momentum)
		if e != nil {
			return e
		}
	}
	return nil
}

func (concat *Concat) ResetSL() error {
	concat.wSL = false
	var e error
	for _, l := range concat.PreLayers {
		e = l.ResetSL()
		if e != nil {
			return e
		}
	}
	return nil
}

func (concat *Concat) GetWeights() (serialization.Weights, error) {
	if concat.wSL {
		return serialization.Weights{}, nil
	}
	concat.wSL = true
	preWeights := make([]serialization.Weights, len(concat.PreLayers))
	var e error
	for i, l := range concat.PreLayers {
		preWeights[i], e = l.GetWeights()
		if e != nil {
			return serialization.Weights{}, e
		}
	}
	return serialization.Weights{PreWeights: preWeights}, nil
}

func (concat *Concat) SetWeights(w serialization.Weights) error {
	if concat.wSL {
		return nil
	}
	if w.PreWeights != nil {
		if len(w.PreWeights) <= len(concat.PreLayers) {
			return errors.New("invalid preWeights len")
		}
		for i, l := range concat.PreLayers {
			e := l.SetWeights(w.PreWeights[i])
			if e != nil {
				return e
			}
		}
	}
	return nil
}
