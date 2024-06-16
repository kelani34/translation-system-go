package main

import (
	"fmt"
	"github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)


func loadModel(modelPath string) (*tensorflow.SavedModel, error){
	model, err := tensorflow.LoadSavedModel(modelPath, []string{"serve"}, nil)
	if err != nil {
		return nil, fmt.Errorf("Error loading model: %v", err)
	}
	return model, nil
}


func main (){

	modelPath := "" // our model path here
	model, err := loadModel(modelPath)


	inputText := "Hello world!"
	
	fmt.Println("Input text: ", inputText)
} 
