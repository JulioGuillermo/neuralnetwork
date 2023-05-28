package layer

import (
	"errors"

	"github.com/julioguillermo/neuralnetwork/pkg/activation"
	"github.com/julioguillermo/neuralnetwork/pkg/serialization"
	"github.com/julioguillermo/neuralnetwork/pkg/tensor"
)

type Deconv2D struct {
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

func NewDeconv2D(filters, kw, kh, stride int, act activation.Activation) *Deconv2D {
	return &Deconv2D{
		OutputShape:  []int{0, 0, filters},
		KernelWidth:  kw,
		KernelHeight: kh,
		Stride:       stride,
		Activation:   act,
		Trainable:    true,
	}
}

func NewInDeconv2D(input_shape []int, filters, kw, kh, stride int, act activation.Activation) *Deconv2D {
	return &Deconv2D{
		InputShape:   input_shape,
		OutputShape:  []int{0, 0, filters},
		KernelWidth:  kw,
		KernelHeight: kh,
		Stride:       stride,
		Activation:   act,
		Trainable:    true,
	}
}

func (deconv *Deconv2D) GetOutShape() []int {
	return deconv.OutputShape
}

func (deconv *Deconv2D) calOutShape() {
	deconv.OutputShape = []int{
		(deconv.InputShape[0]-1)*deconv.Stride + deconv.KernelWidth,
		(deconv.InputShape[1]-1)*deconv.Stride + deconv.KernelHeight,
		deconv.OutputShape[2],
	}
}

func (deconv *Deconv2D) calIndex(p, d int) int {
	return p*deconv.Stride + d
}

func (deconv *Deconv2D) Build() error {
	if deconv.InputShape == nil ||
		len(deconv.InputShape) < 3 ||
		deconv.InputShape[0] < 1 ||
		deconv.InputShape[1] < 1 ||
		deconv.InputShape[2] < 1 {
		return errors.New("invalid input shape")
	}
	if deconv.OutputShape[2] < 1 {
		return errors.New("invalid output shape")
	}
	if deconv.Activation == nil {
		deconv.Activation = &activation.Relu{}
	}
	deconv.calOutShape()
	deconv.PreLayer = nil
	deconv.Weights = tensor.NewWeightTensor(deconv.OutputShape[2], deconv.InputShape[2], deconv.KernelWidth, deconv.KernelHeight)
	deconv.MWeights = tensor.NewZeroTensor(deconv.OutputShape[2], deconv.InputShape[2], deconv.KernelWidth, deconv.KernelHeight)
	deconv.Bias = tensor.NewWeightTensor(deconv.OutputShape...)
	deconv.MBias = tensor.NewZeroTensor(deconv.OutputShape...)
	return nil
}

func (deconv *Deconv2D) SetPrelayer(lay Layer) error {
	if deconv.PreLayer != nil && lay != nil && !tensor.CompareShape(deconv.PreLayer.GetOutShape(), lay.GetOutShape()) {
		return errors.New("invalid prelayers output shape")
	}
	deconv.PreLayer = lay
	return nil
}

func (deconv *Deconv2D) Connect(preLayer Layer) error {
	deconv.InputShape = preLayer.GetOutShape()
	err := deconv.Build()
	if err != nil {
		return err
	}
	deconv.PreLayer = preLayer
	return nil
}

func (deconv *Deconv2D) GetActivation() activation.Activation {
	return deconv.Activation
}

func (deconv *Deconv2D) Reset() error {
	if deconv.cNeta || deconv.cOutput || deconv.cDif != 0 {
		deconv.cNeta = false
		deconv.cOutput = false
		deconv.cDif = 0
		if deconv.PreLayer != nil {
			return deconv.PreLayer.Reset()
		}
	}
	return nil
}

func (deconv *Deconv2D) FullReset() error {
	return deconv.Reset()
}

func (deconv *Deconv2D) GetInput() tensor.Tensor {
	return deconv.input
}

func (deconv *Deconv2D) Get(input tensor.Tensor) (tensor.Tensor, error) {
	if deconv.cNeta {
		return deconv.neta, nil
	}
	if deconv.PreLayer != nil {
		var e error
		input, e = deconv.PreLayer.Output(input)
		if e != nil {
			return nil, e
		}
	}
	return deconv.GetOne(input)
}

func (deconv *Deconv2D) deconvAt(out tensor.Tensor, od, id, x, y int) {
	var w, in float64
	for kx := 0; kx < deconv.KernelWidth; kx++ {
		for ky := 0; ky < deconv.KernelHeight; ky++ {
			in, _ = deconv.input.Get(x, y, id)
			w, _ = deconv.Weights.Get(od, id, kx, ky)
			out.AddAt(in*w, deconv.calIndex(x, kx), deconv.calIndex(y, ky), od)
		}
	}
}

func (deconv *Deconv2D) GetOne(input tensor.Tensor) (tensor.Tensor, error) {
	if deconv.cNeta {
		return deconv.neta, nil
	}
	if len(input.GetShape()) != len(deconv.InputShape) {
		return nil, errors.New("incompatible input shape")
	}
	for i, v := range input.GetShape() {
		if v != deconv.InputShape[i] {
			return nil, errors.New("incompatible input shape")
		}
	}
	deconv.input = input
	out := deconv.Bias.Copy()
	for i := 0; i < deconv.OutputShape[2]; i++ {
		for j := 0; j < deconv.InputShape[2]; j++ {
			for x := 0; x < deconv.InputShape[0]; x++ {
				for y := 0; y < deconv.InputShape[1]; y++ {
					deconv.deconvAt(out, i, j, x, y)
				}
			}
		}
	}
	return out, nil
}

func (deconv *Deconv2D) Output(input tensor.Tensor) (tensor.Tensor, error) {
	if deconv.cOutput {
		return deconv.output, nil
	}
	var err error
	input, err = deconv.Get(input)
	if err != nil {
		return nil, err
	}
	deconv.output, err = deconv.Activation.Activate(input)
	if err != nil {
		return nil, err
	}
	deconv.cOutput = true
	return deconv.output, nil
}

func (deconv *Deconv2D) SetDif(dif tensor.Tensor) {
	dif.Reshape(deconv.OutputShape...)
	if deconv.cDif == 0 {
		deconv.dif = dif
	} else {
		deconv.dif.AddTensor(dif)
	}
	deconv.cDif++
}

func (deconv *Deconv2D) calDifAt(der tensor.Tensor, x, y, od, id int) float64 {
	var w float64
	r := 0.0
	for i := 0; i < deconv.KernelWidth; i++ {
		for j := 0; j < deconv.KernelHeight; j++ {
			w, _ = deconv.Weights.Get(od, id, i, j)
			d, _ := der.Get(deconv.calIndex(x, i), deconv.calIndex(y, j), od)
			r += d * w
		}
	}
	return r
}

func (deconv *Deconv2D) calDif(der tensor.Tensor) tensor.Tensor {
	out := tensor.NewZeroTensor(deconv.InputShape...)
	for od := 0; od < deconv.OutputShape[2]; od++ {
		for id := 0; id < deconv.InputShape[2]; id++ {
			for x := 0; x < deconv.InputShape[0]; x++ {
				for y := 0; y < deconv.InputShape[1]; y++ {
					out.AddAt(deconv.calDifAt(der, x, y, od, id), x, y, id)
				}
			}
		}
	}
	return out
}

func (deconv *Deconv2D) Dif() error {
	if deconv.PreLayer != nil {
		der, err := deconv.PreLayer.GetOne(deconv.PreLayer.GetInput())
		if err != nil {
			return err
		}
		der.Reshape(deconv.InputShape...)
		der, err = deconv.PreLayer.GetActivation().Derive(der)
		if err != nil {
			return err
		}
		out := deconv.calDif(der)
		deconv.PreLayer.SetDif(out)
		err = deconv.PreLayer.Dif()
		if err != nil {
			return err
		}
	}
	return nil
}

func (deconv *Deconv2D) SetTrainable(t bool) {
	deconv.Trainable = t
}

func (deconv *Deconv2D) fitWeight(od, id, i, j int, alpha, momentum float64) error {
	var (
		in float64
		d  float64
		m  float64
		v  float64 = 0
		e  error
	)
	for x := 0; x < deconv.InputShape[0]; x++ {
		for y := 0; y < deconv.InputShape[1]; y++ {
			in, e = deconv.input.Get(x, y, id)
			if e != nil {
				return e
			}
			d, e = deconv.dif.Get(deconv.calIndex(x, i), deconv.calIndex(y, j), od)
			if e != nil {
				return e
			}
			v += in * d * alpha
		}
	}
	m, e = deconv.MWeights.Get(od, id, i, j)
	if e != nil {
		return e
	}
	deconv.Weights.AddAt(v+m*momentum, od, id, i, j)
	deconv.MWeights.Set(v, od, id, i, j)
	return nil
}

func (deconv *Deconv2D) Fit(alpha float64, momentum float64) error {
	if deconv.Trainable {
		var e error
		for od := 0; od < deconv.OutputShape[2]; od++ {
			for id := 0; id < deconv.InputShape[2]; id++ {
				for i := 0; i < deconv.KernelWidth; i++ {
					for j := 0; j < deconv.KernelHeight; j++ {
						e = deconv.fitWeight(od, id, i, j, alpha, momentum)
						if e != nil {
							return e
						}
					}
				}
			}
		}
		m := deconv.dif.Copy()
		deconv.dif.MulNumber(alpha)
		deconv.MBias.MulNumber(momentum)
		deconv.dif.AddTensor(deconv.MBias)
		deconv.Bias.AddTensor(deconv.dif)
		deconv.MBias = m
	}
	if deconv.PreLayer != nil {
		return deconv.PreLayer.Fit(alpha, momentum)
	}
	return nil
}

func (deconv *Deconv2D) ResetSL() error {
	deconv.wSL = false
	if deconv.PreLayer != nil {
		return deconv.PreLayer.ResetSL()
	}
	return nil
}

func (deconv *Deconv2D) GetWeights() (serialization.Weights, error) {
	if deconv.wSL {
		return serialization.Weights{}, nil
	}
	deconv.wSL = true
	data := [][]float64{
		deconv.Weights.GetData(),
		deconv.Bias.GetData(),
	}
	w := serialization.Weights{
		Data: data,
	}
	if deconv.PreLayer != nil {
		pw, e := deconv.PreLayer.GetWeights()
		if e != nil {
			return serialization.Weights{}, e
		}
		w.PreWeights = []serialization.Weights{pw}
	}
	return w, nil
}

func (deconv *Deconv2D) SetWeights(w serialization.Weights) error {
	if !deconv.wSL {
		deconv.wSL = true
		if w.Data != nil {
			deconv.Weights.SetData(w.Data[0])
			deconv.Bias.SetData(w.Data[1])
		}
		if deconv.PreLayer != nil && w.PreWeights != nil {
			if len(w.PreWeights) == 0 {
				return errors.New("invalid preWeights len")
			}
			return deconv.PreLayer.SetWeights(w.PreWeights[0])
		}
	}
	return nil
}
