package serialization

import (
	"encoding/gob"
	"os"
)

func BinSaveWeights(w Weights, path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	enc := gob.NewEncoder(file)
	err = enc.Encode(w)
	return err
}

func BinLoadWeights(path string) (Weights, error) {
	file, err := os.Open(path)
	if err != nil {
		return Weights{}, err
	}
	dec := gob.NewDecoder(file)
	var w Weights
	err = dec.Decode(&w)
	if err != nil {
		return Weights{}, err
	}
	return w, nil
}
