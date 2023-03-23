from transformers import BertModel, BertTokenizer, BertForMaskedLM
from transformers import pipeline


test_sentence_ner = "बनेपाली ले चित्र बनाइरहेका बेला स्थानीय ले गोलो घेरा बनाएर आफ्नो चोक लाई क्यानभास मा हेर्दै थिए । " 

vocab_file_dir = './NepaliBERT/' 
tokenizer = BertTokenizer.from_pretrained(vocab_file_dir,
                                          strip_accents=False,
                                         clean_text=False )


print(dir(tokenizer))
print(tokenizer(test_sentence_ner))

exit()
model = BertForMaskedLM.from_pretrained('./NepaliBERT')

fill_mask = pipeline(
    "fill-mask",
    model=model,
    tokenizer=tokenizer
)


input_nepali_text = 'केही मीठो बात गर, [MASK] त्यसै ढल्किँदै छ'

print(fill_mask(input_nepali_text))
