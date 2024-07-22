import sys
import re

# 原始字符串
def replace_str(text,id,idx):

# 匹配并替换 a 和 b 之间的内容
    id = str(id)
    idx = str(idx)
    result = text.replace(idx + '|',id + '|')
    return result

def read_txt(file_path):
    with open(file_path, 'r') as file:
        res = []
        lines = file.readlines()
        for line in lines:
            res.append(line)
    return res
def write_txt(file_path, data):
    with open(file_path, 'w') as file:
        for line in data:
            file.write(line)
            # file.write('\n')
if __name__ == '__main__':
    file_path = input("label_path:")
    if 'txt' not in file_path:
        file_path = file_path + '/annos.txt'
    lines_replace = read_txt('id_replace.txt')
    lines_label = read_txt(file_path)
    id = 0 # 真实id

    for line_r in lines_replace:

        if len(line_r.split(' ')) == 1:
            id = line_r.split(' ')[0].replace('\n', '')
            if id == '':
                pass
            else:
                id = int(id)
                img_ib = 0 # 起始
                img_ie = 0 # 终止
        elif len(line_r.split(' ')) == 2:
            img_ie = int(line_r.split(' ')[0])
            idx = int(line_r.split(' ')[1])
            for i,line_l in enumerate(lines_label):
                img_i = int(line_l.split(',')[0])
                img_id = int(line_l.split(',')[1].split('|')[0])
                if img_i > img_ib and img_i <= img_ie and idx == img_id:
                    lines_label[i] = replace_str(line_l,id,idx)
            img_ib = img_ie
        else:
            pass
    write_txt(file_path, lines_label)
    # print(lines_label)



