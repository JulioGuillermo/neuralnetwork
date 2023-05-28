package layer

import (
	"errors"

	"github.com/julioguillermo/neuralnetwork/pkg/activation"
	"github.com/julioguillermo/neuralnetwork/pkg/serialization"
	"github.com/julioguillermo/neuralnetwork/pkg/tensor"
)

type Conv2D struct {
	Weights    tensor.Tensor
	MWeights   tensor.Tensor
	Bias       tensor.Tensor
	MBias      tensor.Tensor
	Activation activation.Activation
	PreLayer   Layer

	InputShape   []int
	OutputShape  []int
	KernelWidth  int
	KernelHeight int
	Stride       int

	cNeta   bool
	cOutput bool
	cDif    int
	neta    tensor.Tensor
	output  tensor.Tensor
	input   tensor.Tensor
	dif     tensor.Tensor

	Trainable bool
	wSL       bool
}

func NewConv2D(filters, kw, kh, stride int, act activation.Activation) *Conv2D {
	return &Conv2D{
		OutputShape:  []int{0, 0, filters},
		KernelWidth:  kw,
		KernelHeight: kh,
		Stride:       stride,
		Activation:   act,
		Trainable:    true,
	}
}

func NewInConv2D(input_shape []int, filters, kw, kh, stride int, act activation.Activation) *Conv2D {
	return &Conv2D{
		InputShape:   input_shape,
		OutputShape:  []int{0, 0, filters},
		KernelWidth:  kw,
		KernelHeight: kh,
		Stride:       stride,
		Activation:   act,
		Trainable:    true,
	}
}

func (conv *Conv2D) GetOutShape() []int {
	return conv.OutputShape
}

func (conv *Conv2D) calOutShape() {
	conv.OutputShape = []int{
		(conv.InputShape[0]-conv.KernelWidth)/conv.Stride + 1,
		(conv.InputShape[1]-conv.KernelHeight)/conv.Stride + 1,
		conv.OutputShape[2],
	}
}

func (conv *Conv2D) calIndex(p, w int) int {
	return p*conv.Stride + w
}

func (conv *Conv2D) Build() error {
	if conv.InputShape == nil ||
		len(conv.InputShape) < 3 ||
		conv.InputShape[0] < 1 ||
		conv.InputShape[1] < 1 ||
		conv.InputShape[2] < 1 {
		return errors.New("invalid input shape")
	}
	if conv.OutputShape[2] < 1 {
		return errors.New("invalid output shape")
	}
	if conv.Activation == nil {
		conv.Activation = &activation.Relu{}
	}
	conv.calOutShape()
	conv.PreLayer = nil
	conv.Weights = tensor.NewWeightTensor(conv.OutputShape[2], conv.InputShape[2], conv.KernelWidth, conv.KernelHeight)
	conv.MWeights = tensor.NewZeroTensor(conv.OutputShape[2], conv.InputShape[2], conv.KernelWidth, conv.KernelHeight)
	conv.Bias = tensor.NewWeightTensor(conv.OutputShape...)
	conv.MBias = tensor.NewZeroTensor(conv.OutputShape...)
	return nil
}

func (conv *Conv2D) SetPrelayer(lay Layer) error {
	if conv.PreLayer != nil && lay != nil && !tensor.CompareShape(conv.PreLayer.GetOutShape(), lay.GetOutShape()) {
		return errors.New("invalid prelayers output shape")
	}
	conv.PreLayer = lay
	return nil
}

func (conv *Conv2D) Connect(preLayer Layer) error {
	conv.InputShape = preLayer.GetOutShape()
	err := conv.Build()
	if err != nil {
		return err
	}
	conv.PreLayer = preLayer
	return nil
}

func (conv *Conv2D) GetActivation() activation.Activation {
	return conv.Activation
}

func (conv *Conv2D) Reset() error {
	if conv.cNeta || conv.cOutput || conv.cDif != 0 {
		conv.cNeta = false
		conv.cOutput = false
		conv.cDif = 0
		if conv.PreLayer != nil {
			return conv.PreLayer.Reset()
		}
	}
	return nil
}

func (conv *Conv2D) FullReset() error {
	return conv.Reset()
}

func (conv *Conv2D) GetInput() tensor.Tensor {
	return conv.input
}

func (conv *Conv2D) Get(input tensor.Tensor) (tensor.Tensor, error) {
	if conv.cNeta {
		return conv.neta, nil
	}
	if conv.PreLayer != nil {
		var e error
		input, e = conv.PreLayer.Output(input)
		if e != nil {
			return nil, e
		}
	}
	return conv.GetOne(input)
}

func (conv *Conv2D) convule(od, id, x, y int) float64 {
	var w, in float64
	r := 0.0
	for i := 0; i < conv.KernelWidth; i++ {
		for j := 0; j < conv.KernelHeight; j++ {
			w, _ = conv.Weights.Get(od, id, i, j)
			in, _ = conv.input.Get(conv.calIndex(x, i), conv.calIndex(y, j), id)
			r += w * in
		}
	}
	return r
}

func (conv *Conv2D) GetOne(input tensor.Tensor) (tensor.Tensor, error) {
	if conv.cNeta {
		return conv.neta, nil
	}
	if len(input.GetShape()) != len(conv.InputShape) {
		return nil, errors.New("incompatible input shape")
	}
	for i, v := range input.GetShape() {
		if v != conv.InputShape[i] {
			return nil, errors.New("incompatible input shape")
		}
	}
	conv.input = input
	out := conv.Bias.Copy()
	for i := 0; i < conv.OutputShape[2]; i++ {
		for j := 0; j < conv.InputShape[2]; j++ {
			for x := 0; x < conv.OutputShape[0]; x++ {
				for y := 0; y < conv.OutputShape[1]; y++ {
					out.AddAt(conv.convule(i, j, x, y), x, y, i)
				}
			}
		}
	}
	return out, nil
}

func (conv *Conv2D) Output(input tensor.Tensor) (tensor.Tensor, error) {
	if conv.cOutput {
		return conv.output, nil
	}
	var err error
	input, err = conv.Get(input)
	if err != nil {
		return nil, err
	}
	conv.output, err = conv.Activation.Activate(input)
	if err != nil {
		return nil, err
	}
	conv.cOutput = true
	return conv.output, nil
}

func (conv *Conv2D) SetDif(dif tensor.Tensor) {
	dif.Reshape(conv.OutputShape...)
	if conv.cDif == 0 {
		conv.dif = dif
	} else {
		conv.dif.AddTensor(dif)
	}
	conv.cDif++
}

func (conv *Conv2D) calDifAt(out, der tensor.Tensor, x, y, od int) {
	d, _ := der.Get(x, y, od)
	var w float64
	for id := 0; id < conv.InputShape[2]; id++ {
		for i := 0; i < conv.KernelWidth; i++ {
			for j := 0; j < conv.KernelHeight; j++ {
				w, _ = conv.Weights.Get(od, id, i, j)
				out.AddAt(d*w, conv.calIndex(x, i), conv.calIndex(y, j), id)
			}
		}
	}
}

func (conv *Conv2D) calDif(der tensor.Tensor) tensor.Tensor {
	out := tensor.NewZeroTensor(conv.InputShape...)
	for x := 0; x < conv.OutputShape[0]; x++ {
		for y := 0; y < conv.OutputShape[1]; y++ {
			for od := 0; od < conv.OutputShape[2]; od++ {
				conv.calDifAt(out, der, x, y, od)
			}
		}
	}
	return out
}

func (conv *Conv2D) Dif() error {
	if conv.PreLayer != nil {
		der, err := conv.PreLayer.GetOne(conv.PreLayer.GetInput())
		if err != nil {
			return err
		}
		der.Reshape(conv.InputShape...)
		der, err = conv.PreLayer.GetActivation().Derive(der)
		if err != nil {
			return err
		}
		out := conv.calDif(der)
		conv.PreLayer.SetDif(out)
		err = conv.PreLayer.Dif()
		if err != nil {
			return err
		}
	}
	return nil
}

func (conv *Conv2D) SetTrainable(t bool) {
	conv.Trainable = t
}

func (conv *Conv2D) fitWeight(od, id, i, j int, alpha, momentum float64) error {
	var (
		in float64
		d  float64
		m  float64
		v  float64 = 0
		e  error
	)
	for x := 0; x < conv.OutputShape[0]; x++ {
		for y := 0; y < conv.OutputShape[1]; y++ {
			in, e = conv.input.Get(conv.calIndex(x, i), conv.calIndex(y, j), id)
			if e != nil {
				return e
			}
			d, e = conv.dif.Get(x, y, od)
			if e != nil {
				return e
			}
			v += in * d * alpha
		}
	}
	m, e = conv.MWeights.Get(od, id, i, j)
	if e != nil {
		return e
	}
	conv.Weights.AddAt(v+m*momentum, od, id, i, j)
	conv.MWeights.Set(v, od, id, i, j)
	return nil
}

func (conv *Conv2D) Fit(alpha float64, momentum float64) error {
	if conv.Trainable {
		var e error
		for od := 0; od < conv.OutputShape[2]; od++ {
			for id := 0; id < conv.InputShape[2]; id++ {
				for i := 0; i < conv.KernelWidth; i++ {
					for j := 0; j < conv.KernelHeight; j++ {
						e = conv.fitWeight(od, id, i, j, alpha, momentum)
						if e != nil {
							return e
						}
					}
				}
			}
		}
		m := conv.dif.Copy()
		conv.dif.MulNumber(alpha)
		conv.MBias.MulNumber(momentum)
		conv.dif.AddTensor(conv.MBias)
		conv.Bias.AddTensor(conv.dif)
		conv.MBias = m
	}
	if conv.PreLayer != nil {
		return conv.PreLayer.Fit(alpha, momentum)
	}
	return nil
}

func (conv *Conv2D) ResetSL() error {
	conv.wSL = false
	if conv.PreLayer != nil {
		return conv.PreLayer.ResetSL()
	}
	return nil
}

func (conv *Conv2D) GetWeights() (serialization.Weights, error) {
	if conv.wSL {
		return serialization.Weights{}, nil
	}
	conv.wSL = true
	data := [][]float64{
		conv.Weights.GetData(),
		conv.Bias.GetData(),
	}
	w := serialization.Weights{
		Data: data,
	}
	if conv.PreLayer != nil {
		pw, e := conv.PreLayer.GetWeights()
		if e != nil {
			return serialization.Weights{}, e
		}
		w.PreWeights = []serialization.Weights{pw}
	}
	return w, nil
}

func (conv *Conv2D) SetWeights(w serialization.Weights) error {
	if !conv.wSL {
		conv.wSL = true
		if w.Data != nil {
			conv.Weights.SetData(w.Data[0])
			conv.Bias.SetData(w.Data[1])
		}
		if conv.PreLayer != nil && w.PreWeights != nil {
			if len(w.PreWeights) == 0 {
				return errors.New("invalid preWeights len")
			}
			return conv.PreLayer.SetWeights(w.PreWeights[0])
		}
	}
	return nil
}
