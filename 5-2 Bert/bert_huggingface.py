from transformers import BertTokenizer, BertForQuestionAnswering
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch


model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-mking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# model_cased = AutoModelForQuestionAnswering.from_pretrained("bert-large-cased-whole-word-masking-finetuned-squad")
# tokenizer_cased = AutoTokenizer.from_pretrained("bert-large-cased-whole-word-masking-finetuned-squad")

question = 'How many parameters does BERT-large have?'
answer_text = 'BERT-large is really big.. it has 24-layers and an embedding size of 1,024, for a total of 340M parameters!'

# Apply the tokenizer to the input text, treating them as a text-pair.
input_ids = tokenizer.encode(question, answer_text)
print('The input has a total of {:} tokens.'.format(len(input_ids)))
print(input_ids)

# BERT only need the token IDs, but for the purpose of inspecting the
# tokenizer's behavior, let's also get the token strings and display them.

tokens = tokenizer.convert_ids_to_tokens(input_ids)
print(tokens)

# For each token and its id...
for token, id in zip(tokens, input_ids):
    # If this is the [SEP] token, add some space around it to make it stand out.
    if id == tokenizer.sep_token_id:
        print('')

    # Print the token string and its ID in tow columns.
    print('{:<12} {:>6,}'.format(token, id))
    if id == tokenizer.sep_token_id:
        print('')


# Search the input_ids for the first instance of the '[SEP]' token.
sep_index = input_ids.index(tokenizer.sep_token_id)

# The number of segment A tokens includes the [SEP] token itself.
num_seg_a = sep_index + 1

# The remainder are segment B
num_seg_b = len(input_ids) - num_seg_a

# Construct the list of 0s and 1s
segment_ids = [0]*num_seg_a + [1]*num_seg_b

# There should be a segment_id for every input token.
assert len(segment_ids) == len(input_ids)

# Run our example through the model
start_scores, end_scores = model(torch.tensor([input_ids]), # The tokens representing our input text.
                                 token_type_ids=torch.tensor([segment_ids])) # The segment IDs to differentiate question

# Find the tokens with the highest 'start' and 'end' scores.
answer_start = torch.argmax(start_scores)
answer_end = torch.argmax(end_scores)

# Combine the tokens in the answer and print it out.
answer = ' '.join(tokens[answer_start:answer_end+1])
print('Answer: "' + answer + '"')

# Start with the first token.
answer = tokens[answer_start]

# Select the remaining answer tokens and join them with whitespace.
for i in range(answer_start+1, answer_end+1):

    # If it's a subword token, then recombine it with the previous token.
    if tokens[i][0:2] == '##':
        answer += tokens[i][2:]

    # Otherwise, add a space then the token.
    else:
        answer += ' ' + tokens[i]

print('Answer: "' + answer + '"')


import matplotlib.pyplot as plt
import seaborn as sns

# Use plot styling from seaborn.
sns.set(style='darkgrid')

# Increase the plot size and font size.
# sns.set(font_scale=1.5)
plt.rcParams['figure.figsize'] = (16, 8)

# Pull the scores out of PyTorch Tensors and convert them to 1D numpy arrays.
s_scores = start_scores.detach().numpy().flatten()
e_scores = end_scores.detach().numpy().flatten()

# We'll use the tokens as the x-axis labels. In order to do that, they all need
# to be unique, so we'll add the token index to the end of each one.
token_labels = []
for (i, token) in enumerate(tokens):
    token_labels.append('{:} - {:>2}'.format(token, i))

# Create a barplot showing the start word score for all of the tokens.
ax = sns.barplot(x=token_labels, y=s_scores, ci=None)

# Turn the xlabels vertical.
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center')

# Turn on the vertical grid to help align words to scores.
ax.grid(True)

plt.title('Start Word Scores')
plt.show()

import pandas as pd

# Store the tokens and scores in a DataFrame.
# Each token will have two rows, one for its start score and one for its end
# score. The 'marker' column will differentiate them. A little wacky, I know.
scores = []
for (i, token_label) in enumerate(token_labels):

    # Add the token's start score as one row.
    scores.append({'token_label': token_label,
                   'score': s_scores[i],
                   'marker': 'start'})

    # Add the token's end score as another row.
    scores.append({'token_label': token_label,
                   'score': s_scores[i],
                   'marker': 'end'})

df = pd.DataFrame(scores)

g = sns.catplot(x='token_label', y='score', hue='marker', data=df,
                kind='bar', height=6, aspect=4)

# Turn the xlabels vertical.
g.set_xticklabels(g.ax.get_xticklabels(), rotation=90, ha='center')

# Turn on the vertical grid to help align words to scores.
g.ax.grid(True)

