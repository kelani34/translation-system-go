from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import tensorflow as tf

# Load the pre-trained M2M100 model and tokenizer
model_name = 'facebook/m2m100_418M'
model = M2M100ForConditionalGeneration.from_pretrained(model_name)
tokenizer = M2M100Tokenizer.from_pretrained(model_name)

# Sample input text
input_text = "Hello, world!"
inputs = tokenizer(input_text, return_tensors="tf")

# Define a function to convert the model
@tf.function(input_signature=[tf.TensorSpec([None, None], tf.int32)])
def serving_function(input_ids):
    output = model(input_ids)
    return {"output": output.logits}

# Save the model
tf.saved_model.save(model, "./model", signatures={"serving_default": serving_function})
