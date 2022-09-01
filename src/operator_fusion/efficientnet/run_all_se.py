
import os

repeat, loop = 1, 1
for func in range(9):
  sh_cmd = "../../../release/efficientnet_se_module_main {} {} 0 {}".format(repeat, loop, func)
  os.system(sh_cmd)

