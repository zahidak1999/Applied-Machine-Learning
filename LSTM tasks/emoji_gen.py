import torch
import torch.nn as nn

class WordToEmojiLSTM(nn.Module):
    def __init__(self, encoding_size, output_size):
        super(WordToEmojiLSTM, self).__init__()
        self.lstm = nn.LSTM(encoding_size, 128)
        self.dense = nn.Linear(128, output_size)

    def forward(self, x):
        out, (hidden_state, cell_state) = self.lstm(x)
        return self.dense(hidden_state[-1])

    def loss(self, x, y):
        logits = self.forward(x)
        return nn.functional.cross_entropy(logits, y)

# Complete character encodings
char_encodings = {
    ' ': [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    'h': [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    'a': [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    't': [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    'r': [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    'c': [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
    'f': [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
    'n': [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
    'm': [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
    'l': [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
    'o': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
    'p': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
    's': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]
}
encoding_size = len(char_encodings[' '])

# Mapping of words to emoji
word_to_emoji = {
    "hat ": 0,
    "rat ": 1,
    "cat ": 2,
    "flat": 3,
    "matt": 4,
    "cap ": 5,
    "son ": 6
}
emoji_labels = ['👒', '🐀', '🐱', '🏠', '🧑‍🦱', '🧢', '👦']
output_size = len(emoji_labels)

def word_to_tensor(word):
    return [[char_encodings[char] for char in word]]

words = ["hat ", "rat ", "cat ", "flat", "matt", "cap ", "son "]
x_train = torch.tensor([word_to_tensor(word) for word in words])
y_train = torch.tensor([word_to_emoji[word] for word in words])

x_train = x_train.squeeze(1).permute(1, 0, 2)

model = WordToEmojiLSTM(encoding_size, output_size)

optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
for epoch in range(1000):
    optimizer.zero_grad()
    loss = model.loss(x_train, y_train)
    loss.backward()
    optimizer.step()

    if epoch == 999:
        print('Im done training')

def predict_emoji(model, word):
    x_test = torch.tensor(word_to_tensor(word)).permute(1, 0, 2)
    logits = model(x_test)
    emoji_idx = logits.argmax(1).item()
    return emoji_labels[emoji_idx]

while True:
    word = input("Type a word or 'exit' to exit: ").lower()
    if word == 'exit':
        break
    padded_word = word.ljust(4)[:4]
    try:
        emoji = predict_emoji(model, padded_word)
        print(f"Predicted Emoji: {emoji}")
    except KeyError:
        print("Character not recognized in word.")