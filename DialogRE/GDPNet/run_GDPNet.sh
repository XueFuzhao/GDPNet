export BERT_BASE_DIR=../bert_base

python convert_tf_checkpoint_to_pytorch.py --tf_checkpoint_path=$BERT_BASE_DIR/bert_model.ckpt --bert_config_file=$BERT_BASE_DIR/bert_config.json --pytorch_dump_path=$BERT_BASE_DIR/pytorch_model.bin
python run_classifier.py   --task_name berts  --do_train --do_eval   --data_dir .   --vocab_file $BERT_BASE_DIR/vocab.txt   --bert_config_file $BERT_BASE_DIR/bert_config.json   --init_checkpoint $BERT_BASE_DIR/pytorch_model.bin   --max_seq_length 512   --train_batch_size 24   --learning_rate 3e-5   --num_train_epochs 20.0   --output_dir berts_f1  --gradient_accumulation_steps 6
rm berts_f1/model_best.pt
cp -r berts_f1 berts_f1c
python run_classifier.py   --task_name bertsf1c --do_eval   --data_dir .   --vocab_file $BERT_BASE_DIR/vocab.txt   --bert_config_file $BERT_BASE_DIR/bert_config.json   --init_checkpoint $BERT_BASE_DIR/pytorch_model.bin   --max_seq_length 512   --train_batch_size 24   --learning_rate 3e-5   --num_train_epochs 20.0   --output_dir berts_f1c  --gradient_accumulation_steps 6
python evaluate.py --f1dev berts_f1/logits_dev.txt --f1test berts_f1/logits_test.txt --f1cdev berts_f1c/logits_dev.txt --f1ctest berts_f1c/logits_test.txt

