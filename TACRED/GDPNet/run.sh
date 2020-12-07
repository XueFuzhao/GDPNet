python code/run_tacred.py   --do_train --do_eval   --data_dir tacred   --model spanbert-large-cased   --train_batch_size 32   --eval_batch_size 4   --learning_rate 2e-5   --num_train_epochs 10   --max_seq_length 128   --output_dir GDPNet   --fp16  --gradient_accumulation_steps 32

