import pandas as pd
import sys
sys.path.insert(0,'/Users/ankur.kumar/Desktop/Personal/Projects/open source projects/ALBERT/ALBERT-TF2.0')

from squad_lib import SquadExample, convert_examples_to_features, convert_examples_to_features_v2
import tokenization

def get_start_end_indexes(text, start, end):
    tokens = text.split(' ')
    index = 0
    final_start = -1
    final_end = -1
    for i in range(len(tokens)):
        
        if i == start:
            final_start = index
        elif i == end-1:
            final_end = index + len(tokens[i])
            break
            
        index+= len(tokens[i])+1
    
    return final_start, final_end


if __name__ == "__main__":
    data = pd.read_csv('/Users/ankur.kumar/Desktop/Personal/Projects/open source projects/ALBERT/ALBERT-TF2.0/nq_test/data/train_df.csv')

    index = 0
    qas_id = data.example_id.values[index]
    question_text = data.question_text.values[index]
    paragraph_text = data.document_text.values[index]
    orig_answer_text = ' '.join(data.document_text.values[index].split(' ')[data.long_answer_start.values[index]:data.long_answer_end.values[index]])
    start_position, end_position = get_start_end_indexes(paragraph_text, data.long_answer_start.values[0],data.long_answer_end.values[0])

    example = SquadExample(
        qas_id,
        question_text,
        paragraph_text,
        orig_answer_text=orig_answer_text,
        start_position=start_position,
        end_position=None,
        is_impossible=False,
    )

    spm_model_file = '/Users/ankur.kumar/Desktop/Personal/Projects/open source projects/ALBERT/models/assets/30k-clean.model'
    do_lower_case = True
    tokenizer = tokenization.FullTokenizer(
        vocab_file=None,spm_model_file=spm_model_file, do_lower_case=do_lower_case)

    number_of_examples = convert_examples_to_features_v2(
      examples=[example],
      tokenizer=tokenizer,
      max_seq_length=512,
      doc_stride=128,
      max_query_length=100,
      is_training=True,
      do_lower_case=True)

    