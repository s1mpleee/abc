from sklearn.model_selection import StratifiedKFold

a = []
b = []
a_train = []
a_test = []
b_train = []
b_test = []

file_path = "list.txt"
f = open(file_path)
for line in f:
    a.append(line.strip().split(" ")[0])
    b.append(line[-2])
f.close()

skf = StratifiedKFold(n_splits=10)
skf.get_n_splits(a, b)

print(skf)
for train_index, test_index in skf.split(a, b):
    print(f"train:{train_index} test:{test_index}")


test_txt_file = '/home/s1mple/data/test.txt'
train_txt_file = '/home/s1mple/data/train.txt'

with open(test_txt_file, 'a') as f:
    for index in test_index:
        f.write(f"{a[index]} {b[index]}\n")
f.close()

with open(train_txt_file, 'a') as f:
    for index in train_index:
        f.write(f"{a[index]} {b[index]}\n")
f.close()