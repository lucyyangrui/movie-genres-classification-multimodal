# echo "================ begin ablation ==================="
# echo "================ seq192"
# python train.py --model_name 'mymodel' --max_seq_len 192 --model_save_path './models/MFMGR/save/paper/ablation/' --ablation 'seq128' --save_model_name 'model-seq128.bin'
# echo "================ seq64"
# python train.py --model_name 'mymodel' --max_seq_len 64 --model_save_path './models/MFMGR/save/paper/ablation/' --ablation 'seq64' --save_model_name 'model-seq64.bin'
# echo "================ seq32"
# python train.py --model_name 'mymodel' --max_seq_len 32 --model_save_path './models/MFMGR/save/paper/ablation/' --ablation 'seq32' --save_model_name 'model-seq32.bin'
# 
# echo "================ frame16"
# python train.py --model_name 'mymodel' --max_frame 16 --model_save_path './models/MFMGR/save/paper/ablation/' --ablation 'frame16' --save_model_name 'model-frame16.bin'
# echo "================ frame8"
# python train.py --model_name 'mymodel' --max_frame 8 --model_save_path './models/MFMGR/save/paper/ablation/' --ablation 'frame8' --save_model_name 'model-frame8.bin'
# echo "================ frame4"
# python train.py --model_name 'mymodel' --max_frame 4 --model_save_path './models/MFMGR/save/paper/ablation/' --ablation 'frame4' --save_model_name 'model-frame4.bin'
# 
# echo "================ audio8"
# python train.py --model_name 'mymodel' --audio_part_num 8 --model_save_path './models/MFMGR/save/paper/ablation/' --ablation 'audio8' --save_model_name 'model-audio8.bin'
# echo "================ audio4"
# python train.py --model_name 'mymodel' --audio_part_num 4 --model_save_path './models/MFMGR/save/paper/ablation/' --ablation 'audio8' --save_model_name 'model-audio4.bin'
# 
# echo "================ frame8 audio4"
# python train.py --model_name 'mymodel' --max_frame 8 --audio_part_num 4 --model_save_path './models/MFMGR/save/paper/ablation/' --ablation 'audio8' --save_model_name 'model-frame8-audio4.bin'
# 
# python train.py --model_name 'base1' --max_frame 64 --model_save_path './models/CCT_MMC_base1/' --ablation 'base1-video-audio-test' --save_model_name 'test.bin'
# 
# echo "================ lr"
# pretrain_lr=( 5e-4 5e-3 5e-2 )
# other_lt=( 5e-3 5e-2 )
# for element in ${pretrain_lr[@]}
# do
# echo "================ pre-lr=" $element
# python train.py --model_name 'mymodel' --pretrain_model_lr $element --model_save_path './models/MFMGR/save/paper/ablation/' --ablation 'pre_lr_'$element --save_model_name 'model-pre-lr-'$element'.bin'
# sleep 10s
# done

# for element in ${other_lt[@]}
# do
# echo "================ oth-lr=" $element
# python train.py --model_name 'mymodel' --other_model_lr $element --model_save_path './models/MFMGR/save/paper/ablation/' --ablation 'oth_lr_'$element --save_model_name 'model-oth-lr-'$element'.bin'
# sleep 10s
# done

# ========== goodluck
# echo "two modals"
# python train.py --model_name mymodel --modals summary video --note "summary video" --fusion_size 1
# sleep 10s
# python train.py --model_name mymodel --modals summary audio --note "summary audio" --fusion_size 2
# sleep 10s
# python train.py --model_name mymodel --modals video poster --note "video poster" --fusion_size 2
# sleep 10s
# python train.py --model_name mymodel --modals video audio --note "video audio" --fusion_size 2
# sleep 10s
# python train.py --model_name mymodel --modals poster audio --note "poster audio" --fusion_size 2
# sleep 10s

# echo "three modals"
# python train.py --model_name mymodel --modals summary video poster --note "summary video poster" --fusion_size 2
# sleep 10s
# python train.py --model_name mymodel --modals summary video audio --note "summary video audio" --fusion_size 2
# sleep 10s
# python train.py --model_name mymodel --modals summary poster audio --note "summary poster audio" --fusion_size 3
# sleep 10s
# python train.py --model_name mymodel --modals video poster audio --note "video poster audio" --fusion_size 3
# sleep 10s

python train.py --model_name mymodel --modals video --note "video"
sleep 10s
python train.py --model_name mymodel --modals video poster --note "video poster"
sleep 10s
python train.py --model_name mymodel --modals summary video --note "summary video"
sleep 10s
python train.py --model_name mymodel --modals summary video poster --note "summary video poster"
sleep 10s
python train.py --model_name mymodel --modals summary video poster audio --note "summary video poster audio"
sleep 10s