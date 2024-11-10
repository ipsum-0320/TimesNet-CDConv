export CUDA_VISIBLE_DEVICES=0,1 # 这表示编号为 0 和编号为 1 的 GPU 是可见的。

IS_TRAINING=1
if [ -n "$1" ]; then
  if [ "$1" -eq 0 ] || [ "$1" -eq 1 ]; then
    IS_TRAINING=$1
  fi
fi

PRED_LEN=30
if [ -n "$2" ]; then
  PRED_LEN=$2
fi

PROCESS_DATA=0
if [ -n "$3" ]; then
  if [ "$3" -eq 0 ] || [ "$3" -eq 1 ]; then
    PROCESS_DATA=$3
  fi
fi

SEQ_LEN=360
if [ -n "$4" ]; then
  SEQ_LEN=$4
fi

python -u run.py \
  --task_name long_term_forecast \
  --is_training $IS_TRAINING \
  --root_path ./dataset/TimesNet-CDConv/ \
  --train_data train.csv \
  --test_data test.csv \
  --model_id TimesNet-CDConv \
  --model TimesNet \
  --data custom \
  --features MS \
  --seq_len $SEQ_LEN \
  --label_len $SEQ_LEN \
  --pred_len $PRED_LEN \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 10 \
  --dec_in 10 \
  --c_out 10 \
  --d_model 32 \
  --d_ff 32 \
  --top_k 5 \
  --des Exp \
  --itr 1 \
  --target target \
  --freq t \
  --use_multi_gpu \
  --use_gpu True \
  --process_data $PROCESS_DATA

echo TimesNet-CDConv.sh down