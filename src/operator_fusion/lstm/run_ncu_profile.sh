export HOME=/home/xiachunwei
export TVM_HOME=$HOME/Software/clean_tvm/tvm

export LD_LIBRARY_PATH=$TVM_HOME/build/
export CUDA_HOME="/usr/local/cuda"
export PATH=${CUDA_HOME}/bin/:$PATH
export TORCH_HOME=${HOME}/Software/pytf2.4/lib/python3.7/site-packages/torch
export PYTHONPATH=$TVM_HOME/python:$TORCH_HOME:$PYTHONPATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:${TORCH_HOME}/lib:$LD_LIBRARY_PATH
# sudo -E /home/xiachunwei/anaconda3/bin/python3
sudo -E echo $LD_LIBRARY_PATH
# export CUDA_VISIBLE_DEVICES=1
# sudo HOME=/home/xiachunwei \
#   TVM_HOME=$HOME/Software/clean_tvm/tvm \
#   PYTHONPATH=$TVM_HOME/python:$PYTHONPATH \
#   CUDA_HOME="/usr/local/cuda" \
#   PATH=${CUDA_HOME}/bin/:$PATH \
#   TORCH_HOME="${HOME}/Software/pytf2.4/lib/python3.7/site-packages/torch/lib" \
#   LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:/home/xiachunwei/Software/pytf2.4/lib/python3.7/site-packages/torch/lib:$LD_LIBRARY_PATH \
#   echo ${LD_LIBRARY_PATH} && /usr/local/cuda-11.7/bin/ncu --set full -o lstm_ours -f --target-processes all  /home/xiachunwei/Software/pytf2.4/bin/python3 test_lstm.py

sudo -E /usr/local/cuda-11.7/bin/ncu -k lstm_reuse_shared_memory_v9 --set full -o lstm_ours -f --target-processes all /home/xiachunwei/Software/pytf2.4/bin/python3 test_lstm.py


sudo /usr/local/cuda/bin/ncu --metrics regex:sm__inst_executed* -k fused_sqq_bert_attn -o fused_sqq_bert_attn_opcodes -f --target-processes all ./torch_bert_attn_sqq 1 1 13
