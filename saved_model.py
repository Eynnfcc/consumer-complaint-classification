from transformers import RobertaForSequenceClassification, AutoTokenizer

checkpoint = "./results/checkpoint-48723"

model = RobertaForSequenceClassification.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")

print("Model saved successfully!")