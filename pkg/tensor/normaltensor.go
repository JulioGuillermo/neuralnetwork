package tensor

import (
	"errors"
	"fmt"
)

type NormTensor struct {
	Data   []float64
	Shape  []int
	MShape []int
}

func (self *NormTensor) Size() int {
	return MulIndex(self.Shape, -1)
}

func (self *NormTensor) GetShape() []int {
	return self.Shape
}

func (self *NormTensor) ShapeAt(index int) int {
	return self.Shape[index]
}

func (self *NormTensor) GetData() []float64 {
	return self.Data
}

func (self *NormTensor) SetData(d []float64) {
	if len(self.Data) == len(d) {
		self.Data = d
	} else {
		copy(self.Data, d)
	}
}

func (self *NormTensor) Set(val float64, index ...int) error {
	ind, err := GetRealIndex(index, self.MShape, self.Shape)
	if err != nil {
		return err
	}
	self.Data[ind] = val
	return nil
}

func (self *NormTensor) FSet(val float64, index int) error {
	if index < 0 || index >= len(self.Data) {
		return errors.New("Index out of range.")
	}
	self.Data[index] = val
	return nil
}

func (self *NormTensor) Get(index ...int) (float64, error) {
	ind, err := GetRealIndex(index, self.MShape, self.Shape)
	if err != nil {
		return 0, err
	}
	return self.Data[ind], nil
}

func (self *NormTensor) FGet(index int) (float64, error) {
	if index < 0 || index >= len(self.Data) {
		return 0, errors.New("Index out of range.")
	}
	return self.Data[index], nil
}

func (self *NormTensor) GetSubTensor(index int) (Tensor, error) {
	if index < 0 || index >= self.Shape[0] {
		return nil, errors.New("Index out of range.")
	}
	shape := self.Shape[1:]
	dlen := MulIndex(shape, -1)
	data := make([]float64, dlen)
	offset := self.MShape[0] * index
	copy(data, self.Data[offset:])
	//index = 0
	//for i := offset; i < offset+self.MShape[0]; i++ {
	//	data[index] = self.Data[i]
	//	index++
	//}
	return NewTensor(data, shape...), nil
}

func (self *NormTensor) Run(fun func(float64, int) (float64, error)) error {
	var err error
	for i := 0; i < len(self.Data); i++ {
		self.Data[i], err = fun(self.Data[i], i)
		if err != nil {
			return err
		}
	}
	return nil
}

func (self *NormTensor) Copy() Tensor {
	data := make([]float64, len(self.Data))
	shape := make([]int, len(self.Shape))
	mshape := make([]int, len(self.MShape))
	copy(data, self.Data)
	copy(shape, self.Shape)
	copy(mshape, self.MShape)
	return &NormTensor{
		Data:   data,
		Shape:  shape,
		MShape: mshape,
	}
}

func (self *NormTensor) Reshape(shape ...int) {
	self.Shape = shape
	self.MShape = GetMShape(shape)
	data_len := MulIndex(shape, -1)
	if data_len != len(self.Data) {
		data := make([]float64, data_len)
		copy(data, self.Data)
		self.Data = data
	}
}

func (self *NormTensor) str(index ...int) (string, error) {
	s := "["
	dim := len(index)
	index = append(index, 0)
	if dim == len(self.Shape)-1 {
		var v float64
		var e error
		for i := 0; i < self.Shape[dim]; i++ {
			v, e = self.Get(index...)
			if e != nil {
				return "", e
			}
			s += fmt.Sprint(v)
			if i < self.Shape[dim]-1 {
				s += ", "
			}
			index[dim]++
		}
	} else {
		var v string
		var e error
		for i := 0; i < self.Shape[dim]; i++ {
			v, e = self.str(index...)
			if e != nil {
				return "", e
			}
			s += v
			if i < self.Shape[dim]-1 {
				s += ", "
			}
			index[dim]++
		}
	}
	return s + "]", nil
}

func (self *NormTensor) Str() (string, error) {
	return self.str()
}
