main=/codes/run_vqa_1.py
model=/models/Taiyi-BLIP-full-Chinese
cache=/tmp

model_type=blip

train=/data/VQA/VQA_train.json
valid=/data/VQA/VQA_val.json
test=/data/VQA/VQA_test.json
answer_list=/data/VQA/answer_list.json
predict_result_path=/saves/taiyi_blip-vqa-base-finetuned/result_ch_blip.json

output_dir=/saves/taiyi_blip-vqa-base-finetuned

python $main \
	--output_dir $output_dir \
	--model_name_or_path $model \
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
	--gradient_accumulation_steps="64" \
	--per_device_train_batch_size="4" \
	--per_device_eval_batch_size="4" \
	--evaluation_strategy='epoch' \
	--save_strategy='epoch' \
	--logging_strategy='epoch' \
	--learning_rate="2e-5" --warmup_steps="0" --weight_decay 0.05 \
	--lr_scheduler_type cosine \
	--num_train_epochs="10" \
	--push_to_hub &> logs/taiyi_blip-vqa-base-finetuned.log
