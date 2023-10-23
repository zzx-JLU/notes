import os

# 获取文件名
file_name = input('请输入文件名：')
while not os.path.exists(file_name):
    file_name = input('文件不存在，请重新输入：')

# 新文件
base_name, expand_name = os.path.splitext(file_name)
new_file_name = base_name + '_processed' + '.dat'
new_file_contents = []

# 读取文件并处理
with open(file_name, encoding='utf-8', mode='rt') as file:
    file_contents = file.readlines()
    for line in file_contents:
        begin_idx = line.find('(')
        end_idx = line.find(')')
        content = line[begin_idx: end_idx].strip().strip('()').split(',')
        new_line = '    '.join(content) + '\n'
        new_file_contents.append(new_line)

# 将处理后的数据写入新文件
with open(new_file_name, encoding='utf-8', mode='xt') as file:
    file.writelines(new_file_contents)

print('操作完成')
