package loss

import (
	"math"

	"github.com/julioguillermo/neuralnetwork/pkg/tensor"
)

func CrossEntropy(output, target tensor.Tensor) (tensor.Tensor, error) {
	loss := output.Copy()
	data := target.GetData()
	err := loss.Run(func(o float64, i int) (float64, error) {
		if o == 1 {
			return -math.Log(data[i]), nil
		} else {
			return -math.Log(1 - data[i]), nil
		}
	})
	return loss, err
}
