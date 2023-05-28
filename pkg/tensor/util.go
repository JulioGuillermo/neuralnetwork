package tensor

import (
	"errors"
)

func CompareShape(s1, s2 []int) bool {
	for i, v := range s1 {
		if s2[i] != v {
			return false
		}
	}
	return true
}

func MulIndex(shape []int, index int) int {
	m := 1
	for i := index + 1; i < len(shape); i++ {
		m *= shape[i]
	}
	return m
}

func GetMShape(shape []int) []int {
	shape_len := len(shape)
	mshape := make([]int, shape_len)
	mshape[shape_len-1] = 1
	for i := shape_len - 1; i > 0; i-- {
		mshape[i-1] = mshape[i] * shape[i]
	}
	return mshape
}

func GetRealIndex(index, mshape, shape []int) (int, error) {
	if len(index) != len(mshape) {
		return -1, errors.New("Invalid index dimension.")
	}
	realindex := 0
	for i := 0; i < len(mshape); i++ {
		if index[i] < 0 || index[i] >= shape[i] {
			return -1, errors.New("Index out of range.")
		}
		realindex += index[i] * mshape[i]
	}
	return realindex, nil
}

func ConcatTensors(dim int, tensors ...Tensor) (Tensor, error) {
	ntens := len(tensors)
	if ntens == 1 {
		return tensors[0], nil
	} else if ntens == 0 {
		return nil, nil
	}
	for i := 1; i < ntens; i++ {
		if len(tensors[i].GetShape()) != len(tensors[i-1].GetShape()) {
			return nil, errors.New("Different tensors dimensions.")
		}
	}
	for i := 1; i < ntens; i++ {
		for j := 0; j < len(tensors[i].GetShape()); j++ {
			if j != dim && tensors[i].ShapeAt(j) != tensors[i-1].ShapeAt(j) {
				return nil, errors.New("Incompatible tensors shapes.")
			}
		}
	}
	shape := make([]int, len(tensors[0].GetShape()))
	copy(shape, tensors[0].GetShape())
	for i := 1; i < ntens; i++ {
		shape[dim] += tensors[i].ShapeAt(dim)
	}
	dlen := MulIndex(shape, -1)
	data := make([]float64, dlen)

	out := NewTensor(data, shape...)

	dindex := 0
	var e error
	for i := 0; i < ntens; i++ {
		e = copytensor(out, tensors[i], dim, dindex)
		if e != nil {
			return nil, e
		}
		dindex += tensors[i].ShapeAt(dim)
	}
	return out, nil
}

func copytensor(to, from Tensor, dim, dimIndex int, index ...int) error {
	var e error
	if len(index) == dim {
		index = append(index, dimIndex)
		e = copytensor(to, from, dim, dimIndex, index...)
		if e != nil {
			return e
		}
	} else if len(index) == len(to.GetShape()) {
		fIndex := make([]int, len(index))
		copy(fIndex, index)
		var v float64
		for i := 0; i < from.ShapeAt(dim); i++ {
			fIndex[dim] = i
			v, e = from.Get(fIndex...)
			if e != nil {
				return e
			}
			e = to.Set(v, index...)
			if e != nil {
				return e
			}
			index[dim]++
		}
	} else {
		for i := 0; i < to.ShapeAt(len(index)); i++ {
			e = copytensor(to, from, dim, dimIndex, append(index, i)...)
			if e != nil {
				return e
			}
		}
	}
	return nil
}
