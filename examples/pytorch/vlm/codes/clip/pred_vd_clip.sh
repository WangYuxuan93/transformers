main=/codes/run_vd.py
model=/saves/taiyi_clip-vd-base-finetuned/
cache=/tmp
decoder_model=/models/Taiyi-CLIP-Roberta-102M-Chinese
model_type=clip

train=/data/VD/new_VD_train.json
valid=/data/VD/new_VD_val.json
test=/data/VD/new_VD_test.json
answer_list_path=/data/VD/vd_answer.json
question_list_path=/data/VD/vd_question.json
predict_result_path=/saves/taiyi_clip-vd-base-finetuned/result_ch_clip.json

python $main \
	--model_name_or_path $model \
    --cache_dir $cache \
	--decoder_model_path $decoder_model\
	--model_type $model_type \
	--train_file $train \
	--validation_file $valid \
	--test_file $test \
	--answer_list_file $answer_list_path \
	--question_list_file $question_list_path \
	--predict_result_path $predict_result_path \
	--remove_unused_columns=False \
	--label_names answer_input_ids \
	--do_predict \
	--per_device_eval_batch_size="32" \
