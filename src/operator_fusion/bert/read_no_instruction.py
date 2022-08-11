import os

def read_no_instruction(file_path):
  f = open(file_path, 'r')
  lines = f.readlines()
  no_instructions = []
  for i in range(len(lines)):
    com = lines[i].strip().split("\",")
    if len(com) > 1 and len(com[0]) > 0 and len(com[1]) > 0:
      if(int(com[1]) > 0):
        no_instructions.append((i, com[0], int(com[1])))
  return no_instructions


if __name__=="__main__":
  dir_path = "/home/xiachunwei/Projects/tensor-compiler-gpu/release"
  f1 = os.path.join(dir_path, "fused_sqq_bert_no_instructions.csv")
  f2 = os.path.join(dir_path, "fused_sqq_bert_pipelined_v2_no_instruction.csv")
  inst1 = read_no_instruction(f1)
  inst2 = read_no_instruction(f2)
  print(len(inst1))
  print(len(inst2))
  for i,j,k in inst1:
    if i > 7811:
      print(i, j, k)
  print("-"*100)
  for i,j,k in inst2:
    if i > 7834:
      print(i, j, k)