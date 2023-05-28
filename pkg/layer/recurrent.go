package layer

import (
	"errors"

	"github.com/julioguillermo/neuralnetwork/pkg/activation"
	"github.com/julioguillermo/neuralnetwork/pkg/serialization"
	"github.com/julioguillermo/neuralnetwork/pkg/tensor"
)

type Recurrent struct {
	Weights    tensor.Tensor
	MWeights   tensor.Tensor
	Bias       tensor.Tensor
	MBias      tensor.Tensor
	Activation activation.Activation
	NIn        int
	NOut       int
	PreLayer   Layer

	Trainable bool

	cNeta   bool
	neta    tensor.Tensor
	cOutput bool
	output  tensor.Tensor

	input tensor.Tensor
	dif   tensor.Tensor
	cDif  int

	wSL bool
}

func NewRecurrent(units int, act activation.Activation) *Recurrent {
	return &Recurrent{
		NIn:        0,
		NOut:       units,
		Activation: act,
		Trainable:  true,
	}
}

func NewInRecurrent(inputs, units int, act activation.Activation) *Recurrent {
	return &Recurrent{
		NIn:        inputs,
		NOut:       units,
		Activation: act,
		Trainable:  true,
	}
}

func (recurrent *Recurrent) GetOutShape() []int {
	return []int{recurrent.NOut}
}

func (recurrent *Recurrent) Build() error {
	if recurrent.NIn < 1 {
		return errors.New("invalid input size")
	}
	if recurrent.NOut < 1 {
		return errors.New("invalid output size")
	}
	if recurrent.Activation == nil {
		recurrent.Activation = &activation.Relu{}
	}
	recurrent.NIn += recurrent.NOut
	recurrent.Weights = tensor.NewWeightTensor(recurrent.NOut, recurrent.NIn)
	recurrent.MWeights = tensor.NewZeroTensor(recurrent.NOut, recurrent.NIn)
	recurrent.Bias = tensor.NewWeightTensor(recurrent.NOut)
	recurrent.MBias = tensor.NewZeroTensor(recurrent.NOut)
	recurrent.PreLayer = nil
	return nil
}

func (recurrent *Recurrent) SetPrelayer(lay Layer) error {
	if recurrent.PreLayer != nil && lay != nil && !tensor.CompareShape(recurrent.PreLayer.GetOutShape(), lay.GetOutShape()) {
		return errors.New("invalid prelayers output shape")
	}
	recurrent.PreLayer = lay
	return nil
}

func (recurrent *Recurrent) Connect(preLayer Layer) error {
	inShape := preLayer.GetOutShape()
	recurrent.NIn = 1
	for i := 0; i < len(inShape); i++ {
		recurrent.NIn *= inShape[i]
	}
	err := recurrent.Build()
	if err != nil {
		return err
	}
	recurrent.PreLayer = preLayer
	return nil
}

func (recurrent *Recurrent) GetActivation() activation.Activation {
	return recurrent.Activation
}

func (recurrent *Recurrent) Reset() error {
	if recurrent.cNeta || recurrent.cOutput || recurrent.cDif != 0 {
		recurrent.cNeta = false
		recurrent.cOutput = false
		recurrent.cDif = 0
		if recurrent.PreLayer != nil {
			return recurrent.PreLayer.Reset()
		}
	}
	return nil
}

func (recurrent *Recurrent) FullReset() error {
	recurrent.input = nil
	recurrent.output = nil
	recurrent.dif = nil
	return recurrent.Reset()
}

func (recurrent *Recurrent) GetInput() tensor.Tensor {
	return recurrent.input
}

func (recurrent *Recurrent) Get(input tensor.Tensor) (tensor.Tensor, error) {
	if recurrent.cNeta {
		return recurrent.neta, nil
	}
	if recurrent.PreLayer != nil {
		var err error
		input, err = recurrent.PreLayer.Output(input)
		if err != nil {
			return nil, err
		}
	}
	input = input.Copy()
	input.Reshape(recurrent.NIn)
	if recurrent.output != nil {
		for i := 0; i < recurrent.NOut; i++ {
			v, e := recurrent.output.FGet(i)
			if e != nil {
				return nil, e
			}
			input.FSet(v, recurrent.NIn+i-recurrent.NOut)
		}
	}
	return recurrent.GetOne(input)
}

func (recurrent *Recurrent) GetOne(input tensor.Tensor) (tensor.Tensor, error) {
	if recurrent.cNeta {
		return recurrent.neta, nil
	}
	if input.Size() != recurrent.NIn {
		return nil, errors.New("incompatible input shape")
	}
	recurrent.input = input
	out := recurrent.Bias.Copy()
	var w, in float64
	for i := 0; i < recurrent.NOut; i++ {
		for j := 0; j < recurrent.NIn; j++ {
			w, _ = recurrent.Weights.Get(i, j)
			in, _ = input.FGet(j)
			out.AddAt(w*in, i)
		}
	}
	recurrent.neta = out
	recurrent.cNeta = true
	return out, nil
}

func (recurrent *Recurrent) Output(input tensor.Tensor) (tensor.Tensor, error) {
	if recurrent.cOutput {
		return recurrent.output, nil
	}
	var err error
	recurrent.output, err = recurrent.Get(input)
	if err != nil {
		return nil, err
	}
	recurrent.output, err = recurrent.Activation.Activate(recurrent.output)
	if err != nil {
		return nil, err
	}
	recurrent.cOutput = true
	return recurrent.output, nil
}

func (recurrent *Recurrent) SetDif(dif tensor.Tensor) {
	dif.Reshape(recurrent.NOut)
	if recurrent.cDif == 0 {
		recurrent.dif = dif
	} else {
		recurrent.dif.AddTensor(dif)
	}
	recurrent.cDif++
}

func (recurrent *Recurrent) Dif() error {
	if recurrent.PreLayer != nil {
		der, err := recurrent.PreLayer.GetOne(recurrent.PreLayer.GetInput())
		if err != nil {
			return err
		}
		der.Reshape(recurrent.NIn)
		der, err = recurrent.PreLayer.GetActivation().Derive(der)
		if err != nil {
			return err
		}

		out := tensor.NewZeroTensor(recurrent.NIn)

		var d, w float64
		for i := 0; i < recurrent.Weights.ShapeAt(1); i++ {
			for j := 0; j < recurrent.Weights.ShapeAt(0); j++ {
				d, _ = recurrent.dif.FGet(j)
				w, _ = recurrent.Weights.Get(j, i)
				out.AddAt(d*w, i)
			}
			d, _ = der.FGet(i)
			out.MulAt(d, i)
		}

		recurrent.PreLayer.SetDif(out)
		err = recurrent.PreLayer.Dif()
		if err != nil {
			return err
		}
	}
	return nil
}

func (recurrent *Recurrent) SetTrainable(t bool) {
	recurrent.Trainable = t
}

func (recurrent *Recurrent) Fit(alpha float64, momentum float64) error {
	if recurrent.Trainable {
		var val, v, m float64
		for i := 0; i < recurrent.Weights.ShapeAt(0); i++ {
			m, _ = recurrent.dif.FGet(i)
			val = alpha * m
			for j := 0; j < recurrent.Weights.ShapeAt(1); j++ {
				v, _ = recurrent.input.FGet(j)
				v *= val
				m, _ = recurrent.MWeights.Get(i, j)
				recurrent.Weights.AddAt(v+m*momentum, i, j)
				recurrent.MWeights.Set(v, i, j)
			}
			m, _ = recurrent.MBias.Get(i)
			recurrent.Bias.AddAt(val+m*momentum, i)
			recurrent.MBias.Set(val, i)
		}
	}
	if recurrent.PreLayer != nil {
		return recurrent.PreLayer.Fit(alpha, momentum)
	}
	return nil
}

func (recurrent *Recurrent) ResetSL() error {
	recurrent.wSL = false
	if recurrent.PreLayer != nil {
		return recurrent.PreLayer.ResetSL()
	}
	return nil
}

func (recurrent *Recurrent) GetWeights() (serialization.Weights, error) {
	if recurrent.wSL {
		return serialization.Weights{}, nil
	}
	recurrent.wSL = true
	data := [][]float64{
		recurrent.Weights.GetData(),
		recurrent.Bias.GetData(),
	}

	w := serialization.Weights{
		Data: data,
	}

	if recurrent.PreLayer != nil {
		pw, e := recurrent.PreLayer.GetWeights()
		if e != nil {
			return serialization.Weights{}, e
		}
		w.PreWeights = []serialization.Weights{pw}
	}
	return w, nil
}

func (recurrent *Recurrent) SetWeights(w serialization.Weights) error {
	if !recurrent.wSL {
		recurrent.wSL = true
		if w.Data != nil {
			recurrent.Weights.SetData(w.Data[0])
			recurrent.Bias.SetData(w.Data[1])
		}

		if recurrent.PreLayer != nil && w.PreWeights != nil {
			if len(w.PreWeights) == 0 {
				return errors.New("invalid preWeights len")
			}
			return recurrent.PreLayer.SetWeights(w.PreWeights[0])
		}
	}
	return nil
}
