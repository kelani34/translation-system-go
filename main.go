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



func translate(model *tensorflow.SavedModel, inputText string) (string, error) {
	// Create a tensor from the input text
	inputTensor, err := tensorflow.NewTensor(inputText)
	if err != nil {
		return "", fmt.Errorf("Error creating input tensor: %v", err)
	}

	// Run the model
	output, err := model.Session.Run(
		map[tensorflow.Output]*tensorflow.Tensor{
			model.Graph.Operation("input_tensor_name").Output(0): inputTensor,
		},
		[]tensorflow.Output{
			model.Graph.Operation("output_tensor_name").Output(0),
		},
		nil,
	)
	if err != nil {
		return "", fmt.Errorf("Error running model: %v", err)
	}

	// Convert the output tensor to a string
	translatedText := output[0].Value().(string)

	return translatedText, nil
}



func main (){

	modelPath := "" // our model path here
	model, err := loadModel(modelPath)


	inputText := "Hello world!"
	
	fmt.Println("Input text: ", inputText)
} 
