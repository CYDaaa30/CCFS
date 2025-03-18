CUDA_VISIBLE_DEVICES=3, python /home/yanda/WORK/CCFS/ccfs_cifar10.py \
    --data-path /home/yanda/WORK/Datasets/CIFAR-10/ --filter-model resnet18 --teacher-model resnet18 \
    --teacher-path /home/yanda/WORK/CCFS/checkpoints/resnet18_cifar10_200epochs.pth  --eval-model resnet18 \
    --device cuda --epochs 10 --opt sgd --lr 0.2 --momentum 0.9 --weight-decay 1e-4 \
    --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --lr-warmup-decay 0.01 \
    --distill-data-path /home/yanda/WORK/CCFS/syn-data/cda_cifar10_ipc2000 \
    -T 20 --image-per-class 500 --alpha 0.4 --curriculum-num 3 \
    --select-misclassified --select-method simple --balance \
    --score forgetting --score-path /home/yanda/WORK/CCFS/scores/forgetting_CIFAR10.npy \
    --output-dir /home/yanda/WORK/CCFS/selection_logs --num-eval 2

CUDA_VISIBLE_DEVICES=3, python /home/yanda/WORK/CCFS/eval_cifar10.py \
    --data-path /home/yanda/WORK/Datasets/CIFAR-10/ --eval-model resnet18 \
    --teacher-model resnet18 --teacher-path /home/yanda/WORK/CCFS/checkpoints/resnet18_cifar10_200epochs.pth \
    --device cuda --epochs 10 --opt sgd --lr 0.2 --momentum 0.9 --weight-decay 1e-4 -T 20 \
    --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --lr-warmup-decay 0.01 \
    --distill-data-path /home/yanda/WORK/CCFS/syn-data/cda_cifar10_ipc2000 \
    --selected_indices_path /home/yanda/WORK/CCFS/selection_logs/CIFAR-10/2025-03-18_00-37-46/selected_indices.json \
    --image-per-class 500 --num-eval 2