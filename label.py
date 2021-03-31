import os

def img_file_name(file_dir):
    L = ''
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if file == 'img.png':
                L = os.path.join(root, file)
    #                print(L)
    #                file_name = file[0:-4]  #去掉.png后缀
    #                L.append(file_name)
    #                L.append(' '+'this is anohter file\'s name')
    return L

imgdir = '/media/s1mple/新加卷/project/data/2'
fake_imgdir = 'home/s1mple/data/2'
list_txt_file = 'list.txt'
docs = os.listdir(imgdir)

for name in docs:
    if name.endswith(".jpg"):
        print(name)
        label_folder = imgdir + '/' + name

        with open(list_txt_file, 'a') as f:
            f.write(f"{fake_imgdir}/{name} 1\n")
        f.close()