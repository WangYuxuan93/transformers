main=/codes/run_vqa_1.py
model=/models/chinese-clip-vit-base-patch16-for-vqa 
cache=/tmp
decoder_model=/models/Taiyi-CLIP-Roberta-102M-Chinese 
model_type=chinese

train=/data/VQA/VQA_train.json
valid=/data/VQA/VQA_val.json
test=/data/VQA/VQA_test.json
answer_list=/data/VQA/answer_list.json
predict_result_path=/saves/chinese_clip-vqa-base-finetuned/result_ch_chinese.json

output_dir=/saves/chinese_clip-vqa-base-finetuned

python $main \
	--output_dir $output_dir \
	--model_name_or_path $model \
	--decoder_model_path $decoder_model\
	--model_type $model_type \
    --cache_dir $cache \
	--train_file $train \
	--validation_file $valid \
	--test_file $test \
	--answer_list $answer_list \
	--predict_result_path $predict_result_path \
	--remove_unused_columns=False \
	--label_names answer_input_ids \
	--do_train  \
	--do_eval \
	--do_predict \
	--gradient_accumulation_steps="4" \
	--per_device_train_batch_size="32" \
	--per_device_eval_batch_size="32" \
	--evaluation_strategy='epoch' \
	--save_strategy='epoch' \
	--logging_strategy='epoch' \
	--learning_rate="2e-5" --warmup_steps="0" --weight_decay 0.05 \
	--lr_scheduler_type cosine \
	--num_train_epochs="10" \
	--push_to_hub &> logs/chinese_clip-vqa-base-finetuned.log
