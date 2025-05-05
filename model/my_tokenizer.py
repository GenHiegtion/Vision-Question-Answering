from transformers import BertTokenizer

def load_bert_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer

tokenizer = load_bert_tokenizer()

PAD_ID = tokenizer.pad_token_id
UNK_ID = tokenizer.unk_token_id
CLS_ID = tokenizer.cls_token_id
SEP_ID = tokenizer.sep_token_id

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

def detokenize(sequence):
    filtered_sequence = [idx for idx in sequence
                        if idx != PAD_ID and idx != CLS_ID and idx != SEP_ID]

    return tokenizer.decode(filtered_sequence, skip_special_tokens=True)