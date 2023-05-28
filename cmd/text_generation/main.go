package main

import (
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"strings"

	"github.com/julioguillermo/neuralnetwork/pkg/activation"
	"github.com/julioguillermo/neuralnetwork/pkg/layer"
	"github.com/julioguillermo/neuralnetwork/pkg/loss"
	"github.com/julioguillermo/neuralnetwork/pkg/model"
	"github.com/julioguillermo/neuralnetwork/pkg/serialization"
	"github.com/julioguillermo/neuralnetwork/pkg/tensor"
)

const (
	Symbols  = string(1) + "abcdefghijklmnopqrstuvwxyz0123456789 "
	InSize   = len(Symbols)
	Epochs   = 100
	Alpha    = 0.001
	Momentum = 0.5
)

func LoadDS() []string {
	bytes, err := os.ReadFile("text_gen.json")
	if err != nil {
		panic(err)
	}

	var data map[string]interface{}
	json.Unmarshal(bytes, &data)
	rows, ok := data["rows"].([]interface{})
	if !ok {
		panic("invalid row type")
	}

	targets := []string{}
	for _, r := range rows {
		ro, ok := r.(map[string]interface{})
		if !ok {
			panic("invalid row type...")
		}
		row, ok := ro["row"].(map[string]interface{})
		if !ok {
			panic("invalid row type...")
		}
		target, ok := row["target"].(string)
		if !ok {
			panic("target is not string")
		}

		target = strings.TrimRight(target, ".")
		target = strings.TrimRight(target, " ")
		targets = append(targets, strings.ToLower(target))
	}
	return targets
}

func CharToTensor(c byte) tensor.Tensor {
	p := 0
	for i, s := range Symbols {
		if byte(s) == c {
			p = i
		}
	}
	t := tensor.NewZeroTensor(InSize)
	t.Set(1, p)
	return t
}

func CharFromTensor(t tensor.Tensor) byte {
	return byte(Symbols[t.MaxIndex()])
}

func GetModel() model.Model {
	m := model.NewSequential()
	m.AddLayer(layer.NewInDense(InSize, 10, activation.NewTanh()))
	m.AddLayer(layer.NewRecurrent2(30, activation.NewTanh()))
	m.AddLayer(layer.NewRecurrent2(30, activation.NewTanh()))
	m.AddLayer(layer.NewDense(InSize, activation.NewSigmoid()))
	return m
}

func TrainTarget(m model.Model, t string, it, max, e, es int) {
	lt := 0.0

	for i := 0; i < len(t)-1; i++ {
		l, _ := m.TrainOne(CharToTensor(t[i]), CharToTensor(t[i+1]), Alpha, Momentum, loss.L1)
		lt += l.Abs().Sum() / float64(l.Size()) / float64(len(t))
	}

	m.TrainOne(CharToTensor(t[len(t)-1]), tensor.NewZeroTensor(InSize), Alpha, Momentum, loss.L1)
	fmt.Printf("\r[%d / %d] <%d / %d> => %f", e, es, it, max, lt)
}

func OneTrain(m model.Model, ts []string, e, es int) {
	max := len(ts)
	for i, t := range ts {
		m.FullReset()
		TrainTarget(m, t, i, max, e, es)
	}
}

func Train(m model.Model, ds []string) {
	for i := 0; i < Epochs; i++ {
		OneTrain(m, ds, i, Epochs)
		w, _ := m.GetModelWeights()
		serialization.BinSaveWeights(w, "model.bin")
	}
	fmt.Println()
}

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	ds := LoadDS()
	m := GetModel()
	w, e := serialization.BinLoadWeights("model.bin")
	if e == nil {
		m.SetModelWeights(w)
	}

	var s string
	var o tensor.Tensor
	for {
		fmt.Print(">>> ")
		fmt.Scanln(&s)
		if s == "exit" {
			break
		}
		if s == "train" {
			Train(m, ds)
			continue
		}
		s = strings.Replace(s, "_", " ", -1)
		m.FullReset()
		for j := 0; j < len(s)-1; j++ {
			m.Predict(CharToTensor(s[j]))
		}
		o = CharToTensor(s[len(s)-1])
		s = ""
		for i := 0; i < 50; i++ {
			o, _ = m.Predict(o)
			/*if o.Max() < 0.3 || o.MaxIndex() == 0 {
				break
			}*/
			s += string(CharFromTensor(o))
		}
		fmt.Println(" => ", s)
	}
}
