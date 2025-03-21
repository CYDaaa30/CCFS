# ipc = 50
CUDA_VISIBLE_DEVICES=0, python ccfs_tiny.py \
    --data-path /path/to/Tiny-ImageNet/ --filter-model resnet18 --teacher-model resnet18 \
    --teacher-path ./checkpoints/resnet18_tiny_200epochs.pth  --eval-model resnet18 \
    --device cuda --batch-size 64 --epochs 100 --opt sgd --lr 0.2 --momentum 0.9 --weight-decay 1e-4 \
    --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --lr-warmup-decay 0.01 \
    --distill-data-path ./syn-data/cda_tiny_rn18_4k_ipc100 \
    -T 20 --image-per-class 50 --alpha 0.2 --curriculum-num 3 \
    --select-misclassified --select-method simple --balance \
    --score forgetting --score-path ./scores/forgetting_Tiny.npy \
    --output-dir ./selection_logs --num-eval 5
# ipc = 100
CUDA_VISIBLE_DEVICES=0, python ccfs_tiny.py \
    --data-path /path/to/Tiny-ImageNet/ --filter-model resnet18 --teacher-model resnet18 \
    --teacher-path ./checkpoints/resnet18_tiny_200epochs.pth  --eval-model resnet18 \
    --device cuda --batch-size 64 --epochs 100 --opt sgd --lr 0.2 --momentum 0.9 --weight-decay 1e-4 \
    --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --lr-warmup-decay 0.01 \
    --distill-data-path ./syn-data/cda_tiny_rn18_4k_ipc100 \
    -T 20 --image-per-class 100 --alpha 0.2 --curriculum-num 3 \
    --select-misclassified --select-method simple --balance \
    --score forgetting --score-path ./scores/forgetting_Tiny.npy \
    --output-dir ./selection_logs --num-eval 5