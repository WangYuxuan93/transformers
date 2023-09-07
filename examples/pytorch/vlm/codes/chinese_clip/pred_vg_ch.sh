main=/codes/run_vg.py
model=/saves/chinese_clip-vg-base-finetuned/
cache=/tmp
decoder_model=/models/Taiyi-CLIP-Roberta-102M-Chinese
model_type=chinese

train=/data/VG/VG_train.json
valid=/data/VG/VG_val.json
test=/data/VG/VG_test.json
predict_result_path=/saves/chinese_clip-vg-base-finetuned/result_ch_chinese.json

python $main \
	--model_name_or_path $model \
    --decoder_model_path $decoder_model\
    --cache_dir $cache \
	--model_type $model_type \
	--train_file $train \
	--validation_file $valid \
	--test_file $test \
	--remove_unused_columns=False \
	--label_names target_bbox \
	--do_predict \
	--predict_result_path $predict_result_path \
	--per_device_eval_batch_size="64" \
