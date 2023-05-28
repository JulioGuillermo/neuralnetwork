package main

import (
	"fmt"
	"runtime"
	"sync"
)

var wg sync.WaitGroup

func print(ind int) {
	for i := 0; i < 10000; i++ {
		fmt.Println(ind, "=>", i)
	}
	wg.Done()
}

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())

	wg.Add(100 * 10000)

	for i := 0; i < 100; i++ {
		go print(i)
	}
	wg.Wait()
}
