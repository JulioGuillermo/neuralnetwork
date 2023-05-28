package tensor

import "math/rand"

func NewTensor(data []float64, shape ...int) Tensor {
	return &NormTensor{
		Data:   data,
		Shape:  shape,
		MShape: GetMShape(shape),
	}
}

func NewZeroTensor(shape ...int) Tensor {
	data_len := MulIndex(shape, -1)
	data := make([]float64, data_len)
	for i := 0; i < data_len; i++ {
		data[i] = 0
	}
	return NewTensor(data, shape...)
}

func NewOneTensor(shape ...int) Tensor {
	data_len := MulIndex(shape, -1)
	data := make([]float64, data_len)
	for i := 0; i < data_len; i++ {
		data[i] = 1
	}
	return NewTensor(data, shape...)
}

func NewRandTensor(min float64, max float64, shape ...int) Tensor {
	data_len := MulIndex(shape, -1)
	data := make([]float64, data_len)
	scale := max - min
	for i := 0; i < data_len; i++ {
		data[i] = rand.Float64()*scale + min
	}
	return NewTensor(data, shape...)
}

func NewWeightTensor(shape ...int) Tensor {
	return NewRandTensor(-1, 1, shape...)
}

func NewNormTensor(shape ...int) Tensor {
	return NewRandTensor(0, 1, shape...)
}
