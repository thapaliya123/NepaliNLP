from datasets import load_dataset

test_data_path = '/home/fm-pc-lt-125/Documents/personal/Nepali_NLP/NepaliNLP/hugging_face/EverestNER-test-bio.txt'
ner_test_data = load_dataset("text", data_files = test_data_path, field='data')

print(ner_test_data['train'][:5])