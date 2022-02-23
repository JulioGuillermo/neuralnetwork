package serialization

import (
	"encoding/json"
	"io/ioutil"
	"os"
)

func JsonSaveWeights(w Weights, path string) error {
	bytes, err := json.Marshal(w)
	if err != nil {
		return err
	}

	err = ioutil.WriteFile(path, bytes, 0644)
	if err != nil {
		return err
	}

	return nil
}

func JsonLoadWeights(path string) (Weights, error) {
	file, err := os.Open(path)
	if err != nil {
		return Weights{}, err
	}

	decoder := json.NewDecoder(file)
	if err != nil {
		return Weights{}, err
	}

	var w Weights
	err = decoder.Decode(&w)
	if err != nil {
		return Weights{}, err
	}

	return w, nil
}
