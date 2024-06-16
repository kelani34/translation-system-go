from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import tensorflow as tf

model_name = 'facebook/m2m100_418M'
model = M2M100ForConditionalGeneration.from_pretrained(model_name)
tokenizer = M2M100Tokenizer.from_pretrained(model_name)

# Sample input
input_text = "Hello, world!"
inputs = tokenizer(input_text, return_tensors="tf")

# Convert the model
tf_model = tf.saved_model.save(model, "./")
