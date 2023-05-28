package tensor

type Tensor interface {
	Size() int
	GetShape() []int
	ShapeAt(index int) int

	GetData() []float64
	SetData([]float64)

	Set(val float64, index ...int) error
	Get(index ...int) (float64, error)

	FSet(val float64, index int) error
	FGet(index int) (float64, error)

	GetSubTensor(index int) (Tensor, error)

	Run(func(float64, int) (float64, error)) error
	Copy() Tensor
	Reshape(index ...int)

	//Math
	AddAt(val float64, index ...int) error
	AddNumber(val float64) error
	AddTensor(val Tensor) error

	SubAt(val float64, index ...int) error
	SubNumber(val float64) error
	SubTensor(val Tensor) error

	MulAt(val float64, index ...int) error
	MulNumber(val float64) error
	MulTensor(val Tensor) error

	DivAt(val float64, index ...int) error
	DivNumber(val float64) error
	DivTensor(val Tensor) error

	RemAt(val float64, index ...int) error
	RemNumber(val float64) error
	RemTensor(val Tensor) error

	PowAt(val float64, index ...int) error
	PowNumber(val float64) error
	PowTensor(val Tensor) error

	DotProduct(val Tensor) (float64, error)

	Inc()
	Dec()

	Abs() Tensor
	Sign() Tensor
	Sum() float64

	Max() float64
	MaxIndex() int
	Min() float64
	MinIndex() int

	Str() (string, error)
}
