package layer

import (
	"errors"

	"github.com/julioguillermo/neuralnetwork/pkg/activation"
	"github.com/julioguillermo/neuralnetwork/pkg/serialization"
	"github.com/julioguillermo/neuralnetwork/pkg/tensor"
)

type Dense struct {
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

func NewDense(units int, act activation.Activation) *Dense {
	return &Dense{
		NIn:        0,
		NOut:       units,
		Activation: act,
		Trainable:  true,
	}
}

func NewInDense(inputs, units int, act activation.Activation) *Dense {
	return &Dense{
		NIn:        inputs,
		NOut:       units,
		Activation: act,
		Trainable:  true,
	}
}

func (dense *Dense) GetOutShape() []int {
	return []int{dense.NOut}
}

func (dense *Dense) Build() error {
	if dense.NIn < 1 {
		return errors.New("invalid input size")
	}
	if dense.NOut < 1 {
		return errors.New("invalid output size")
	}
	if dense.Activation == nil {
		dense.Activation = &activation.Relu{}
	}
	dense.Weights = tensor.NewWeightTensor(dense.NOut, dense.NIn)
	dense.MWeights = tensor.NewZeroTensor(dense.NOut, dense.NIn)
	dense.Bias = tensor.NewWeightTensor(dense.NOut)
	dense.MBias = tensor.NewZeroTensor(dense.NOut)
	dense.PreLayer = nil
	return nil
}

func (dense *Dense) SetPrelayer(lay Layer) error {
	if dense.PreLayer != nil && lay != nil && !tensor.CompareShape(dense.PreLayer.GetOutShape(), lay.GetOutShape()) {
		return errors.New("invalid prelayers output shape")
	}
	dense.PreLayer = lay
	return nil
}

func (dense *Dense) Connect(preLayer Layer) error {
	inShape := preLayer.GetOutShape()
	dense.NIn = 1
	for i := 0; i < len(inShape); i++ {
		dense.NIn *= inShape[i]
	}
	err := dense.Build()
	if err != nil {
		return err
	}
	dense.PreLayer = preLayer
	return nil
}

func (dense *Dense) GetActivation() activation.Activation {
	return dense.Activation
}

func (dense *Dense) Reset() error {
	if dense.cNeta || dense.cOutput || dense.cDif != 0 {
		dense.cNeta = false
		dense.cOutput = false
		dense.cDif = 0
		if dense.PreLayer != nil {
			return dense.PreLayer.Reset()
		}
	}
	return nil
}

func (dense *Dense) FullReset() error {
	return dense.Reset()
}

func (dense *Dense) GetInput() tensor.Tensor {
	return dense.input
}

func (dense *Dense) Get(input tensor.Tensor) (tensor.Tensor, error) {
	if dense.cNeta {
		return dense.neta, nil
	}
	if dense.PreLayer != nil {
		var err error
		input, err = dense.PreLayer.Output(input)
		if err != nil {
			return nil, err
		}
	}
	return dense.GetOne(input)
}

func (dense *Dense) GetOne(input tensor.Tensor) (tensor.Tensor, error) {
	if dense.cNeta {
		return dense.neta, nil
	}
	if input.Size() != dense.NIn {
		return nil, errors.New("incompatible input shape")
	}
	input.Reshape(dense.NIn)
	dense.input = input
	out := dense.Bias.Copy()
	var w, in float64
	for i := 0; i < dense.NOut; i++ {
		for j := 0; j < dense.NIn; j++ {
			w, _ = dense.Weights.Get(i, j)
			in, _ = input.FGet(j)
			out.AddAt(w*in, i)
		}
	}
	dense.neta = out
	dense.cNeta = true
	return out, nil
}

func (dense *Dense) Output(input tensor.Tensor) (tensor.Tensor, error) {
	if dense.cOutput {
		return dense.output, nil
	}
	var err error
	input, err = dense.Get(input)
	if err != nil {
		return nil, err
	}
	dense.output, err = dense.Activation.Activate(input)
	if err != nil {
		return nil, err
	}
	dense.cOutput = true
	return dense.output, nil
}

func (dense *Dense) SetDif(dif tensor.Tensor) {
	dif.Reshape(dense.NOut)
	if dense.cDif == 0 {
		dense.dif = dif
	} else {
		dense.dif.AddTensor(dif)
	}
	dense.cDif++
}

func (dense *Dense) Dif() error {
	if dense.PreLayer != nil {
		der, err := dense.PreLayer.GetOne(dense.PreLayer.GetInput())
		if err != nil {
			return err
		}
		der.Reshape(dense.NIn)
		der, err = dense.PreLayer.GetActivation().Derive(der)
		if err != nil {
			return err
		}

		out := tensor.NewZeroTensor(dense.NIn)

		var d, w float64
		for i := 0; i < dense.Weights.ShapeAt(1); i++ {
			for j := 0; j < dense.Weights.ShapeAt(0); j++ {
				d, _ = dense.dif.FGet(j)
				w, _ = dense.Weights.Get(j, i)
				out.AddAt(d*w, i)
			}
			d, _ = der.FGet(i)
			out.MulAt(d, i)
		}

		dense.PreLayer.SetDif(out)
		err = dense.PreLayer.Dif()
		if err != nil {
			return err
		}
	}
	return nil
}

func (dense *Dense) SetTrainable(t bool) {
	dense.Trainable = t
}

func (dense *Dense) Fit(alpha float64, momentum float64) error {
	if dense.Trainable {
		var val, v, m float64
		for i := 0; i < dense.Weights.ShapeAt(0); i++ {
			m, _ = dense.dif.FGet(i)
			val = alpha * m
			for j := 0; j < dense.Weights.ShapeAt(1); j++ {
				v, _ = dense.input.FGet(j)
				v *= val
				m, _ = dense.MWeights.Get(i, j)
				dense.Weights.AddAt(v+m*momentum, i, j)
				dense.MWeights.Set(v, i, j)
			}
			m, _ = dense.MBias.Get(i)
			dense.Bias.AddAt(val+m*momentum, i)
			dense.MBias.Set(val, i)
		}
	}
	if dense.PreLayer != nil {
		return dense.PreLayer.Fit(alpha, momentum)
	}
	return nil
}

func (dense *Dense) ResetSL() error {
	dense.wSL = false
	if dense.PreLayer != nil {
		return dense.PreLayer.ResetSL()
	}
	return nil
}

func (dense *Dense) GetWeights() (serialization.Weights, error) {
	if dense.wSL {
		return serialization.Weights{}, nil
	}
	dense.wSL = true
	data := [][]float64{
		dense.Weights.GetData(),
		dense.Bias.GetData(),
	}

	w := serialization.Weights{
		Data: data,
	}

	if dense.PreLayer != nil {
		pw, e := dense.PreLayer.GetWeights()
		if e != nil {
			return serialization.Weights{}, e
		}
		w.PreWeights = []serialization.Weights{pw}
	}
	return w, nil
}

func (dense *Dense) SetWeights(w serialization.Weights) error {
	if !dense.wSL {
		dense.wSL = true
		if w.Data != nil {
			dense.Weights.SetData(w.Data[0])
			dense.Bias.SetData(w.Data[1])
		}

		if dense.PreLayer != nil && w.PreWeights != nil {
			if len(w.PreWeights) == 0 {
				return errors.New("invalid preWeights len")
			}
			return dense.PreLayer.SetWeights(w.PreWeights[0])
		}
	}
	return nil
}
