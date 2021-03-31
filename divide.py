from sklearn.model_selection import StratifiedKFold

a = []
b = []

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
    a_train, a_test = a[train_index], a[test_index]
    b_train, b_test = b[train_index], b[test_index]
