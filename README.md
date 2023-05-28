# NeuralNetwork

This is a simple neural network library inspired by TensorFlow.

It hasn't computational graph, for that it's great for beginer who want to learn how neural networks work.

## Features

### Model

- Sequential

### Layers

- Concat
- Conv2D
- Deconv2D
- Dense
- Flatten
- Input
- Join
- Maxpool2D
- Recurrent
- Recurrent2
- Reshape
- Subtensor

### Activation

- Linear
- Null
- LeakyRelu
- Relu
- Sigmoid
- Tanh
- Sin

### Serialization

- Binary
- JSON

### Loss

- BinaryCrossEntropy
- CrossEntropy
- KullbackLeibler
- L1
- L2
- L3

## Example

```go
func getDS() ([]tensor.Tensor, []tensor.Tensor) {
    fmt.Println("Creating DS...")
    // Inputs
    x := []tensor.Tensor{
        tensor.NewTensor([]float64{0, 0}, 2),
        tensor.NewTensor([]float64{0, 1}, 2),
        tensor.NewTensor([]float64{1, 0}, 2),
        tensor.NewTensor([]float64{1, 1}, 2),
    }
    // Outputs
    y := []tensor.Tensor{
        tensor.NewTensor([]float64{0}, 1),
        tensor.NewTensor([]float64{1}, 1),
        tensor.NewTensor([]float64{1}, 1),
        tensor.NewTensor([]float64{0}, 1),
    }
    return x, y
}

func getModel() model.Model {
    fmt.Println("Creating Model...")
    // New sequential model
    // A sequential is a simple model to put layers
    // The sequential model itself can be use as a layer too
    m := model.NewSequential()
    // Make a dense layer (Full connected layer) with sigmoid activation
    // This will be also the input layer
    m.AddLayer(layer.NewInDense(2, 3, activation.NewSigmoid()))
    // Make an other dense layer
    m.AddLayer(layer.NewDense(5, activation.NewSigmoid()))
    // Make an other dense layer
    m.AddLayer(layer.NewDense(1, activation.NewSigmoid()))
    return m
}

func train(m model.Model, x, y []tensor.Tensor) {
    fmt.Println("Training...")
    // Train the model using the given data set
    // x are inputs and y are the targets
    // learning rate: 0.01
    // momentum: 0.5
    // epochs: 1000
    // batch: 0 (0 => not use batch)
    // verbose: 1 (can be 0 no verbose, 1 basic, 2 full)
    // loss function: L1 (work better in must of the case)
    // shuffle the data set: false
    m.Train(x, y, 0.01, 0.5, 1000, 0, 1, loss.L1, false)
    // save model weights after train
    w, e := m.GetModelWeights()
    if e != nil {
        fmt.Println(e)
    }
    // save them using binary format
    e = serialization.BinSaveWeights(w, "model.gob")
    if e != nil {
        fmt.Println(e)
    }
}

func load(m model.Model) error {
    fmt.Println("Loading...")
    // load model weights using binary format
    w, e := serialization.BinLoadWeights("model.gob")
    if e != nil {
        return e
    }
    // set model weights
    e = m.SetModelWeights(w)
    if e != nil {
        return e
    }
    return nil
}

func test(m model.Model, x, y []tensor.Tensor) {
    fmt.Println("Testing...")
    for i, in := range x {
        o, _ := m.Predict(in)
        os, _ := o.Str()
        xs, _ := in.Str()
        ys, _ := y[i].Str()
        fmt.Printf("\033[34m%s\033[0m : \033[33m%s\033[0m => \033[32m%s\033[0m\n", xs, os, ys)
    }
}

func main() {
    // get dataset
    x, y := getDS()
    // get model
    m := getModel()
    // test the model without training
    test(m, x, y)

    // train the model
    train(m, x, y)

    // load model from file
    /*
    e := load(m)
    if e != nil {
        train(m, x, y)
    }*/

    // test the model after train
    test(m, x, y)
}
```
