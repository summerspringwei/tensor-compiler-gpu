import xlrd

file_path = "/home2/xiachunwei/Projects/商丘/2023级商丘学院学号.xlsx"
book = xlrd.open_workbook(file_path)
sh = book.sheet_by_index(0)
print("{0} {1} {2}".format(sh.name, sh.nrows, sh.ncols))
group_size = 12
data_groups = []
start = 1
name_idx = 3
stu_name = set()
while start < sh.nrows:
    name = sh.cell_value(start, name_idx)
    print(name)
    start += 1
    stu_name.add(str(name))

import os
folder = "/home2/xiachunwei/Projects/商丘/23商丘学院"
subdirs = [x[0] for x in os.walk(folder)]
print(subdirs)
dir_name_set = set()
for dir_name in subdirs:
    dir_name = os.path.basename(os.path.normpath(dir_name))
    print(dir_name)
    if dir_name.find("+") > 0:
      name, id_card = str(dir_name).split("+")
      dir_name_set.add(str(name))

print(dir_name_set)
print('--')
for name in stu_name:
   if name not in dir_name_set:
      print(name[1:])
