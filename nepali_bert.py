from transformers import BertModel, BertTokenizer, BertForMaskedLM
from transformers import pipeline

vocab_file_dir = './NepaliBERT/' 
tokenizer = BertTokenizer.from_pretrained(vocab_file_dir,
                                          strip_accents=False,
                                         clean_text=False )

model = BertForMaskedLM.from_pretrained('./NepaliBERT')

fill_mask = pipeline(
    "fill-mask",
    model=model,
    tokenizer=tokenizer
)


input_nepali_text = 'केही मीठो बात गर, [MASK] त्यसै ढल्किँदै छ'

print(fill_mask(input_nepali_text))