from transformers import BertTokenizer

def load_bert_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer

tokenizer = load_bert_tokenizer()

def tokenize(question, max_seq_len):
    encoding = tokenizer.encode_plus(
        question,
        add_special_tokens=True,
        max_length=max_seq_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    return encoding['input_ids'].squeeze(0)