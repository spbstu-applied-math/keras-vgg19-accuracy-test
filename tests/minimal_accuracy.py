import json

metrics = json.load(open('data/eval.json'))


assert metrics['accuracy'] >= 0.5
assert metrics['top_5_accuracy'] >= 0.7

print('Ok')