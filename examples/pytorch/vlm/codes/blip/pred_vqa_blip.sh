main=/codes/run_vqa.py
model=/saves/taiyi_blip-vqa-base-finetuned/
cache=/tmp

model_type=blip

train=/data/VQA/VQA_train.json
valid=/data/VQA/VQA_val.json
test=/data//VQA/VQA_test.json
answer_list=/data/VQA/answer_list.json
predict_result_path=/saves/taiyi_blip-vqa-base-finetuned/result_ch_blip.json

python $main \
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
	--do_predict \
	--per_device_eval_batch_size="64" \
