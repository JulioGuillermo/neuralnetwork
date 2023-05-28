package serialization

type Weights struct {
	Data       [][]float64 `json:data`
	PreWeights []Weights   `json:pre_weights`
}
