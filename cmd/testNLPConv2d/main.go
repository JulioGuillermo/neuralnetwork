package main

import "github.com/julioguillermo/neuralnetwork/pkg/tensor"

func charToTensor(c byte) tensor.Tensor {
	t := tensor.NewZeroTensor(1, 255)
	t.Set(1, 0, int(c))
	return t
}

func stringToTensor(s string) tensor.Tensor {
	t := tensor.NewZeroTensor(0)
	for _, c := range s {
		t, _ = tensor.ConcatTensors(0, t, charToTensor(byte(c)))
	}
	return t
}

func main() {
	/*es := []string{
		"hola",
		"como estas",
		"estoy bien",
		"carro",
	}
	en := []string{
		"hello",
		"how are you",
		"i am fine",
		"car",
	}*/

}
