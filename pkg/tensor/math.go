package tensor

import (
	"errors"
	"math"
)

// Add
func (self *NormTensor) AddAt(val float64, index ...int) error {
	d, e := self.Get(index...)
	if e != nil {
		return e
	}
	d += val
	return self.Set(d, index...)
}

func (self *NormTensor) AddNumber(val float64) error {
	for i := 0; i < len(self.Data); i++ {
		self.Data[i] += val
	}
	return nil
}

func (self *NormTensor) AddTensor(val Tensor) error {
	val_shape := val.GetShape()
	if len(self.Shape) != len(val_shape) {
		return errors.New("Invalid shape dimensions.")
	}
	for i := 0; i < len(self.Shape); i++ {
		if self.Shape[i] != val_shape[i] {
			return errors.New("Invalid shapes.")
		}
	}
	var e float64
	for i := 0; i < len(self.Data); i++ {
		e, _ = val.FGet(i)
		self.Data[i] += e
	}
	return nil
}

// Sub
func (self *NormTensor) SubAt(val float64, index ...int) error {
	d, e := self.Get(index...)
	if e != nil {
		return e
	}
	d -= val
	return self.Set(d, index...)
}

func (self *NormTensor) SubNumber(val float64) error {
	for i := 0; i < len(self.Data); i++ {
		self.Data[i] -= val
	}
	return nil
}

func (self *NormTensor) SubTensor(val Tensor) error {
	val_shape := val.GetShape()
	if len(self.Shape) != len(val_shape) {
		return errors.New("Invalid shape dimensions.")
	}
	for i := 0; i < len(self.Shape); i++ {
		if self.Shape[i] != val_shape[i] {
			return errors.New("Invalid shapes.")
		}
	}
	var e float64
	for i := 0; i < len(self.Data); i++ {
		e, _ = val.FGet(i)
		self.Data[i] -= e
	}
	return nil
}

// Mul
func (self *NormTensor) MulAt(val float64, index ...int) error {
	d, e := self.Get(index...)
	if e != nil {
		return e
	}
	d *= val
	return self.Set(d, index...)
}

func (self *NormTensor) MulNumber(val float64) error {
	for i := 0; i < len(self.Data); i++ {
		self.Data[i] *= val
	}
	return nil
}

func (self *NormTensor) MulTensor(val Tensor) error {
	val_shape := val.GetShape()
	if len(self.Shape) != len(val_shape) {
		return errors.New("Invalid shape dimensions.")
	}
	for i := 0; i < len(self.Shape); i++ {
		if self.Shape[i] != val_shape[i] {
			return errors.New("Invalid shapes.")
		}
	}
	var e float64
	for i := 0; i < len(self.Data); i++ {
		e, _ = val.FGet(i)
		self.Data[i] *= e
	}
	return nil
}

// Div
func (self *NormTensor) DivAt(val float64, index ...int) error {
	if val == 0 {
		return errors.New("Division by zero")
	}
	d, e := self.Get(index...)
	if e != nil {
		return e
	}
	d /= val
	return self.Set(d, index...)
}

func (self *NormTensor) DivNumber(val float64) error {
	if val == 0.0 {
		return errors.New("Division by zero.")
	}
	for i := 0; i < len(self.Data); i++ {
		self.Data[i] /= val
	}
	return nil
}

func (self *NormTensor) DivTensor(val Tensor) error {
	val_shape := val.GetShape()
	if len(self.Shape) != len(val_shape) {
		return errors.New("Invalid shape dimensions.")
	}
	for i := 0; i < len(self.Shape); i++ {
		if self.Shape[i] != val_shape[i] {
			return errors.New("Invalid shapes.")
		}
	}
	var e float64
	for i := 0; i < len(self.Data); i++ {
		e, _ = val.FGet(i)
		if e == 0.0 {
			return errors.New("Division by zero.")
		}
		self.Data[i] /= e
	}
	return nil
}

// Rem
func (self *NormTensor) RemAt(val float64, index ...int) error {
	if val == 0 {
		return errors.New("Division by zero")
	}
	d, e := self.Get(index...)
	if e != nil {
		return e
	}
	d = float64(int(d) % int(val))
	return self.Set(d, index...)
}

func (self *NormTensor) RemNumber(val float64) error {
	if val == 0.0 {
		return errors.New("Division by zero.")
	}
	for i := 0; i < len(self.Data); i++ {
		self.Data[i] = float64(int(self.Data[i]) % int(val))
	}
	return nil
}

func (self *NormTensor) RemTensor(val Tensor) error {
	val_shape := val.GetShape()
	if len(self.Shape) != len(val_shape) {
		return errors.New("Invalid shape dimensions.")
	}
	for i := 0; i < len(self.Shape); i++ {
		if self.Shape[i] != val_shape[i] {
			return errors.New("Invalid shapes.")
		}
	}
	var e float64
	for i := 0; i < len(self.Data); i++ {
		e, _ = val.FGet(i)
		if e == 0.0 {
			return errors.New("Division by zero.")
		}
		self.Data[i] = float64(int(self.Data[i]) % int(e))
	}
	return nil
}

// Pow
func (self *NormTensor) PowAt(val float64, index ...int) error {
	d, e := self.Get(index...)
	if e != nil {
		return e
	}
	d = math.Pow(d, val)
	return self.Set(d, index...)
}

func (self *NormTensor) PowNumber(val float64) error {
	for i := 0; i < len(self.Data); i++ {
		self.Data[i] = math.Pow(self.Data[i], val)
	}
	return nil
}

func (self *NormTensor) PowTensor(val Tensor) error {
	val_shape := val.GetShape()
	if len(self.Shape) != len(val_shape) {
		return errors.New("Invalid shape dimensions.")
	}
	for i := 0; i < len(self.Shape); i++ {
		if self.Shape[i] != val_shape[i] {
			return errors.New("Invalid shapes.")
		}
	}
	var e float64
	for i := 0; i < len(self.Data); i++ {
		e, _ = val.FGet(i)
		self.Data[i] = math.Pow(self.Data[i], e)
	}
	return nil
}

// Scalar Product
func (self *NormTensor) DotProduct(val Tensor) (float64, error) {
	data := val.GetData()
	if len(data) != len(self.Data) {
		return 0, errors.New("Incompatible tensors sizes.")
	}
	sp := 0.0
	for i, v := range self.Data {
		sp += v * data[i]
	}
	return sp, nil
}

// Inc Dec
func (self *NormTensor) Inc() {
	for i := 0; i < len(self.Data); i++ {
		self.Data[i]++
	}
}

func (self *NormTensor) Dec() {
	for i := 0; i < len(self.Data); i++ {
		self.Data[i]--
	}
}

// Abs
func (self *NormTensor) Abs() Tensor {
	size := len(self.Data)
	data := make([]float64, size)
	copy(data, self.Data)
	for i := 0; i < size; i++ {
		if data[i] < 0 {
			data[i] = -data[i]
		}
	}
	return NewTensor(data, self.Shape...)
}

// Sign
func (self *NormTensor) Sign() Tensor {
	size := len(self.Data)
	data := make([]float64, size)
	copy(data, self.Data)
	for i := 0; i < size; i++ {
		if data[i] < 0 {
			data[i] = -1
		} else {
			data[i] = 1
		}
	}
	return NewTensor(data, self.Shape...)
}

// Sum
func (self *NormTensor) Sum() float64 {
	sum := 0.0
	for i := 0; i < len(self.Data); i++ {
		sum += self.Data[i]
	}
	return sum
}

// Max Min
func (self *NormTensor) Max() float64 {
	max := 0.0
	for i := 0; i < len(self.Data); i++ {
		if max < self.Data[i] {
			max = self.Data[i]
		}
	}
	return max
}

func (self *NormTensor) MaxIndex() int {
	max := 0.0
	pos := 0
	for i := 0; i < len(self.Data); i++ {
		if max < self.Data[i] {
			max = self.Data[i]
			pos = i
		}
	}
	return pos
}

func (self *NormTensor) Min() float64 {
	max := 0.0
	for i := 0; i < len(self.Data); i++ {
		if max > self.Data[i] {
			max = self.Data[i]
		}
	}
	return max
}

func (self *NormTensor) MinIndex() int {
	max := 0.0
	pos := 0
	for i := 0; i < len(self.Data); i++ {
		if max > self.Data[i] {
			max = self.Data[i]
			pos = i
		}
	}
	return pos
}
