package loss

import (
	"math"

	"github.com/julioguillermo/neuralnetwork/pkg/tensor"
)

func KullbackLeibler(output, target tensor.Tensor) (tensor.Tensor, error) {
	loss := target.Copy()

	err := loss.SubTensor(output)
	if err != nil {
		return nil, err
	}

	sign := loss.Sign()

	tdata := target.GetData()
	odata := output.GetData()
	err = loss.Run(func(o float64, i int) (float64, error) {
		return tdata[i] * math.Log(tdata[i]/odata[i]), nil
	})
	if err != nil {
		return nil, err
	}

	err = loss.MulTensor(sign)
	if err != nil {
		return nil, err
	}
	return loss, nil
}
