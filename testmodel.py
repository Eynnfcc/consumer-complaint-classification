from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

checkpoint = "./results/checkpoint-48723"

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")

print("Saved successfully!")