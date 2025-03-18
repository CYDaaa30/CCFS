CUDA_VISIBLE_DEVICES=3, python /home/yanda/WORK/CCFS/ccfs_tiny.py \
    --data-path /home/yanda/WORK/Datasets/Tiny-ImageNet/ --filter-model resnet18 --teacher-model resnet18 \
    --teacher-path /home/yanda/WORK/CCFS/checkpoints/resnet18_tiny_200epochs.pth  --eval-model resnet18 \
    --device cuda --batch-size 64 --epochs 100 --opt sgd --lr 0.2 --momentum 0.9 --weight-decay 1e-4 \
    --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --lr-warmup-decay 0.01 \
    --distill-data-path /home/yanda/WORK/CCFS/syn-data/cda_tiny_rn18_ipc200 \
    -T 20 --image-per-class 50 --alpha 0.2 --curriculum-num 3 \
    --select-misclassified --select-method simple --balance \
    --score forgetting --score-path /home/yanda/WORK/CCFS/scores/forgetting_Tiny.npy \
    --output-dir /home/yanda/WORK/CCFS/selection_logs --num-eval 2

CUDA_VISIBLE_DEVICES=3, python /home/yanda/WORK/CCFS/eval_tiny.py \
    --data-path /home/yanda/WORK/Datasets/Tiny-ImageNet/ --eval-model resnet18 \
    --teacher-model resnet18 --teacher-path /home/yanda/WORK/CCFS/checkpoints/resnet18_tiny_200epochs.pth \
    --device cuda --batch-size 64 --epochs 10 --opt sgd --lr 0.2 --momentum 0.9 --weight-decay 1e-4 -T 20 \
    --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --lr-warmup-decay 0.01 \
    --distill-data-path /home/yanda/WORK/CCFS/syn-data/cda_tiny_rn18_ipc200 \
    --selected_indices_path /home/yanda/WORK/CCFS/selection_logs/Tiny/2025-03-17_23-50-51/selected_indices.json \
    --image-per-class 50 --num-eval 2