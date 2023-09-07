main=/codes/run_new_vg.py
model=/saves/taiyi_blip-vg-base-finetuned/
cache=/tmp
model_type=blip

train=/data/VG/VG_train.json
test=/data/VG/VG_val.json
valid=/data/VG/VG_test.json
predict_result_path=/saves/taiyi_blip-vg-base-finetuned/result_ch_blip.json

python $main \
	--model_name_or_path $model \
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
