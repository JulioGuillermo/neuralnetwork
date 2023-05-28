package layer

import (
	"errors"

	"github.com/julioguillermo/neuralnetwork/pkg/activation"
	"github.com/julioguillermo/neuralnetwork/pkg/serialization"
	"github.com/julioguillermo/neuralnetwork/pkg/tensor"
)

type Join struct {
	PreLayers []Layer
	Shape     []int

	output  tensor.Tensor
	der     tensor.Tensor
	dif     tensor.Tensor
	cOutput bool
	cDer    bool
	cDif    int

	wSL bool
}

func NewJoin(layers ...Layer) (*Join, error) {
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
			if i != 0 && s != tmp[i] {
				return nil, errors.New("incompatible layers outputs shape")
			}
		}
		shape[0] += tmp[0]
	}

	return &Join{
		PreLayers: layers,
		Shape:     shape,
	}, nil
}

func (join *Join) GetOutShape() []int {
	return join.Shape
}

func (join *Join) Build() error {
	return errors.New("this layer can not be used as model input")
}

func (join *Join) SetPrelayer(lay Layer) error {
	if lay == nil {
		return nil
	}
	return errors.New("invalid prelayer change")
}

func (join *Join) Connect(p Layer) error {
	return nil
}

func (join *Join) GetActivation() activation.Activation {
	return activation.ActNull
}

func (join *Join) Reset() error {
	if join.cOutput || join.cDif != 0 {
		join.cOutput = false
		join.cDif = 0
		var e error
		for _, l := range join.PreLayers {
			e = l.Reset()
			if e != nil {
				return e
			}
		}
	}
	return nil
}

func (join *Join) FullReset() error {
	return join.Reset()
}

func (join *Join) GetInput() tensor.Tensor {
	return join.output
}

func (join *Join) Get(input tensor.Tensor) (tensor.Tensor, error) {
	if join.cOutput {
		return join.output, nil
	}
	outs := make([]tensor.Tensor, len(join.PreLayers))
	var e error
	for i, l := range join.PreLayers {
		outs[i], e = l.Output(input)
		if e != nil {
			return nil, e
		}
	}
	join.output, e = tensor.ConcatTensors(0, outs...)
	join.output.Reshape(join.Shape...)
	join.cOutput = true
	return join.output, e
}

func (join *Join) GetOne(input tensor.Tensor) (tensor.Tensor, error) {
	if join.cDer {
		return join.der, nil
	}
	outs := make([]tensor.Tensor, len(join.PreLayers))
	var e error
	for i, l := range join.PreLayers {
		outs[i], e = l.GetOne(l.GetInput())
		if e != nil {
			return nil, e
		}
		outs[i], e = l.GetActivation().Derive(outs[i])
		if e != nil {
			return nil, e
		}
	}
	join.der, e = tensor.ConcatTensors(0, outs...)
	join.cDer = true
	return join.der, e
}

func (join *Join) Output(input tensor.Tensor) (tensor.Tensor, error) {
	return join.Get(input)
}

func (join *Join) SetDif(dif tensor.Tensor) {
	dif.Reshape(join.Shape...)
	if join.cDif == 0 {
		join.dif = dif
	} else {
		join.dif.AddTensor(dif)
	}
	join.cDif++
}

func (join *Join) Dif() error {
	var t tensor.Tensor
	var e error
	st := 0
	if len(join.Shape) == 1 {
		for _, l := range join.PreLayers {
			size := l.GetOutShape()[0]
			data := join.dif.GetData()[st : st+size]
			l.SetDif(tensor.NewTensor(data, size))
			e = l.Dif()
			if e != nil {
				return nil
			}
		}
	} else {
		for _, l := range join.PreLayers {
			ntens := l.GetOutShape()[0]
			tens := make([]tensor.Tensor, ntens)
			for i := 0; i < ntens; i++ {
				t, e = join.dif.GetSubTensor(st)
				st++
				if e != nil {
					return e
				}
				t.Reshape(append([]int{1}, t.GetShape()...)...)
				tens[i] = t
			}
			ten, e := tensor.ConcatTensors(0, tens...)
			if e != nil {
				return e
			}
			l.SetDif(ten)
			e = l.Dif()
			if e != nil {
				return nil
			}
		}
	}
	return nil
}

func (join *Join) SetTrainable(bool) {}

func (join *Join) Fit(alpha, momentum float64) error {
	var e error
	for _, l := range join.PreLayers {
		e = l.Fit(alpha, momentum)
		if e != nil {
			return e
		}
	}
	return nil
}

func (join *Join) ResetSL() error {
	join.wSL = false
	var e error
	for _, l := range join.PreLayers {
		e = l.ResetSL()
		if e != nil {
			return e
		}
	}
	return nil
}

func (join *Join) GetWeights() (serialization.Weights, error) {
	if join.wSL {
		return serialization.Weights{}, nil
	}
	join.wSL = true
	preWeights := make([]serialization.Weights, len(join.PreLayers))
	var e error
	for i, l := range join.PreLayers {
		preWeights[i], e = l.GetWeights()
		if e != nil {
			return serialization.Weights{}, e
		}
	}
	return serialization.Weights{PreWeights: preWeights}, nil
}

func (join *Join) SetWeights(w serialization.Weights) error {
	if join.wSL {
		return nil
	}
	if w.PreWeights != nil {
		if len(w.PreWeights) <= len(join.PreLayers) {
			return errors.New("invalid preWeights len")
		}
		for i, l := range join.PreLayers {
			e := l.SetWeights(w.PreWeights[i])
			if e != nil {
				return e
			}
		}
	}
	return nil
}
