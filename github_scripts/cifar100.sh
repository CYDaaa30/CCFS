CUDA_VISIBLE_DEVICES=3, python /home/yanda/WORK/CCFS/ccfs_cifar100.py \
    --data-path /home/yanda/WORK/Datasets/CIFAR-100/ --filter-model resnet18 --teacher-model resnet18 \
    --teacher-path /home/yanda/WORK/CCFS/checkpoints/resnet18_cifar100_200epochs.pth  --eval-model resnet18 \
    --device cuda --epochs 10 --opt sgd --lr 0.2 --momentum 0.9 --weight-decay 1e-4 \
    --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --lr-warmup-decay 0.01 \
    --distill-data-path /home/yanda/WORK/CCFS/syn-data/cda_cifar100_ipc200 \
    -T 20 --image-per-class 25 --alpha 0.2 --curriculum-num 3 \
    --select-misclassified --select-method simple --balance \
    --score forgetting --score-path /home/yanda/WORK/CCFS/scores/forgetting_CIFAR100.npy \
    --output-dir /home/yanda/WORK/CCFS/selection_logs --num-eval 2

CUDA_VISIBLE_DEVICES=3, python /home/yanda/WORK/CCFS/eval_cifar100.py \
    --data-path /home/yanda/WORK/Datasets/CIFAR-100/ --eval-model resnet18 \
    --teacher-model resnet18 --teacher-path /home/yanda/WORK/CCFS/checkpoints/resnet18_cifar100_200epochs.pth \
    --device cuda --epochs 10 --opt sgd --lr 0.2 --momentum 0.9 --weight-decay 1e-4 -T 20 \
    --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --lr-warmup-decay 0.01 \
    --distill-data-path /home/yanda/WORK/CCFS/syn-data/cda_cifar100_ipc200 \
    --selected_indices_path /home/yanda/WORK/CCFS/selection_logs/CIFAR-100/2025-03-18_00-22-54/selected_indices.json \
    --image-per-class 25 --num-eval 2