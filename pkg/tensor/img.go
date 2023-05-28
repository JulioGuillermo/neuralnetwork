package tensor

import (
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"os"
)

// Img to tensor
func TensorFromRGBAImg(img image.Image) Tensor {
	bounds := img.Bounds()
	w := bounds.Max.X - bounds.Min.X
	h := bounds.Max.Y - bounds.Min.Y
	tens := NewZeroTensor(w, h, 4)
	for i := 0; i < w; i++ {
		for j := 0; j < h; j++ {
			c := img.At(i+bounds.Min.X, j+bounds.Min.Y)

			r, g, b, a := c.RGBA()
			tens.Set(float64(r)/255.0, i, j, 0)
			tens.Set(float64(g)/255.0, i, j, 1)
			tens.Set(float64(b)/255.0, i, j, 2)
			tens.Set(float64(a)/255.0, i, j, 3)
		}
	}
	return tens
}

func TensorFromRGBImg(img image.Image) Tensor {
	bounds := img.Bounds()
	w := bounds.Max.X - bounds.Min.X
	h := bounds.Max.Y - bounds.Min.Y
	tens := NewZeroTensor(w, h, 3)
	for i := 0; i < w; i++ {
		for j := 0; j < h; j++ {
			c := img.At(i+bounds.Min.X, j+bounds.Min.Y)

			r, g, b, _ := c.RGBA()
			tens.Set(float64(r)/255.0, i, j, 0)
			tens.Set(float64(g)/255.0, i, j, 1)
			tens.Set(float64(b)/255.0, i, j, 2)
		}
	}
	return tens
}

func TensorFromGrayImg(img image.Image) Tensor {
	bounds := img.Bounds()
	w := bounds.Max.X - bounds.Min.X
	h := bounds.Max.Y - bounds.Min.Y
	tens := NewZeroTensor(w, h, 1)
	for i := 0; i < w; i++ {
		for j := 0; j < h; j++ {
			c := img.At(i+bounds.Min.X, j+bounds.Min.Y)

			g := color.GrayModel.Convert(c).(color.Gray)
			tens.Set(float64(g.Y)/255.0, i, j, 0)
		}
	}
	return tens
}

func TensorFromJPEG(path string) (Tensor, error) {
	imgfile, e := os.Open(path)
	defer imgfile.Close()
	if e != nil {
		return nil, e
	}
	img, e := jpeg.Decode(imgfile)
	if e != nil {
		return nil, e
	}
	return TensorFromRGBImg(img), nil
}

func TensorFromGrayJPEG(path string) (Tensor, error) {
	imgfile, e := os.Open(path)
	defer imgfile.Close()
	if e != nil {
		return nil, e
	}
	img, e := jpeg.Decode(imgfile)
	if e != nil {
		return nil, e
	}
	return TensorFromGrayImg(img), nil
}

func TensorFromPNG(path string) (Tensor, error) {
	imgfile, e := os.Open(path)
	defer imgfile.Close()
	if e != nil {
		return nil, e
	}
	img, e := png.Decode(imgfile)
	if e != nil {
		return nil, e
	}
	return TensorFromRGBAImg(img), nil
}

func TensorFromGrayPNG(path string) (Tensor, error) {
	imgfile, e := os.Open(path)
	defer imgfile.Close()
	if e != nil {
		return nil, e
	}
	img, e := png.Decode(imgfile)
	if e != nil {
		return nil, e
	}
	return TensorFromGrayImg(img), nil
}

// Tensor to img
func GetTensorRGBAColor(tens Tensor, x, y int) color.Color {
	var (
		r float64
		g float64
		b float64
		a float64
	)
	if len(tens.GetShape()) == 2 {
		r, _ = tens.Get(x, y)
		g, _ = tens.Get(x, y)
		b, _ = tens.Get(x, y)
		a, _ = tens.Get(x, y)
	} else if tens.ShapeAt(2) == 1 {
		r, _ = tens.Get(x, y, 0)
		g, _ = tens.Get(x, y, 0)
		b, _ = tens.Get(x, y, 0)
		a = 1
	} else if tens.ShapeAt(2) == 2 {
		r, _ = tens.Get(x, y, 0)
		g, _ = tens.Get(x, y, 0)
		b, _ = tens.Get(x, y, 0)
		a, _ = tens.Get(x, y, 1)
	} else if tens.ShapeAt(2) == 3 {
		r, _ = tens.Get(x, y, 0)
		g, _ = tens.Get(x, y, 1)
		b, _ = tens.Get(x, y, 2)
		a = 1
	} else {
		r, _ = tens.Get(x, y, 0)
		g, _ = tens.Get(x, y, 1)
		b, _ = tens.Get(x, y, 2)
		a, _ = tens.Get(x, y, 3)
	}
	return color.RGBA{
		R: uint8(r * 255),
		G: uint8(g * 255),
		B: uint8(b * 255),
		A: uint8(a * 255),
	}
}

func GetTensorGrayColor(tens Tensor, x, y int) color.Color {
	var (
		c float64
	)
	if len(tens.GetShape()) == 2 {
		c, _ = tens.Get(x, y)
	} else if tens.ShapeAt(2) == 3 || tens.ShapeAt(2) == 4 {
		r, _ := tens.Get(x, y, 0)
		g, _ := tens.Get(x, y, 1)
		b, _ := tens.Get(x, y, 2)
		c = (r + g + b) / 3.0
	} else {
		c, _ = tens.Get(x, y, 0)
	}
	return color.Gray{
		Y: uint8(c * 255),
	}
}

func TensorToRGBAImg(tensor Tensor) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, tensor.ShapeAt(0), tensor.ShapeAt(1)))
	for i := 0; i < tensor.ShapeAt(0); i++ {
		for j := 0; j < tensor.ShapeAt(1); j++ {
			img.Set(i, j, GetTensorRGBAColor(tensor, i, j))
		}
	}
	return img
}

func TensorToGrayImg(tensor Tensor) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, tensor.ShapeAt(0), tensor.ShapeAt(1)))
	for i := 0; i < tensor.ShapeAt(0); i++ {
		for j := 0; j < tensor.ShapeAt(1); j++ {
			img.Set(i, j, GetTensorGrayColor(tensor, i, j))
		}
	}
	return img
}

func SaveTensorAsPNG(tensor Tensor, path string) error {
	img := TensorToRGBAImg(tensor)
	out, e := os.Create(path)
	defer out.Close()
	if e != nil {
		return e
	}
	return png.Encode(out, img)
}

func SaveTensorAsJPEG(tensor Tensor, path string) error {
	img := TensorToRGBAImg(tensor)
	out, e := os.Create(path)
	defer out.Close()
	if e != nil {
		return e
	}
	return jpeg.Encode(out, img, &jpeg.Options{Quality: 90})
}
