main=/codes/run_vqa.py
model=/saves/taiyi_clip-vqa-base-finetuned/


cache=/tmp
decoder_model=/models/Taiyi-CLIP-Roberta-102M-Chinese

model_type=clip

train=/data/VQA/VQA_train.json
valid=/data/VQA/VQA_val.json
test=/data/VQA/VQA_test.json
answer_list=/data/VQA/answer_list.json
predict_result_path=/saves/taiyi_clip-vqa-base-finetuned/result_ch_clip.json

python $main \
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
	--do_predict \
	--per_device_eval_batch_size="64" \
