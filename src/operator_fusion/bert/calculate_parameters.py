def compute_bert_weight(max_seq_length, num_heads, hidden_size, intermedia_size, num_layers):
  max_tmp_output_size = 0
  weight_sum = 0
  # attn qkv matmul
  weight_sum += (3* num_heads*hidden_size * num_heads*hidden_size)
  max_tmp_output_size = max(max_tmp_output_size, max_seq_length * 3 * num_heads * hidden_size)
  max_tmp_output_size = max(max_tmp_output_size, num_heads * max_seq_length * max_seq_length)
  max_tmp_output_size = max(max_tmp_output_size, num_heads * max_seq_length * hidden_size)
  # attn query-key-value fc
  weight_sum += (num_heads*hidden_size * num_heads*hidden_size)
  # FC1
  weight_sum += (num_heads*hidden_size * intermedia_size)
  max_tmp_output_size = max(max_tmp_output_size, max_seq_length * intermedia_size)
  # FC2
  weight_sum += (num_heads*hidden_size * intermedia_size)
  max_tmp_output_size = max(max_tmp_output_size, max_seq_length * num_heads * hidden_size)

  return (weight_sum * num_heads) / 1024 / 1024, max_tmp_output_size / 1024


if __name__=="__main__":
  # print(compute_bert_weight(512, 16, 64, 8192, 24))
  print(compute_bert_weight(384, 12, 64, 3072, 24))
  
