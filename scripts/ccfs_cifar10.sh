# ipc = 250
CUDA_VISIBLE_DEVICES=0, python /home/yanda/WORK/CCFS/ccfs_cifar10.py \
    --data-path /path/to/CIFAR-10/ --filter-model resnet18 --teacher-model resnet18 \
    --teacher-path ./checkpoints/resnet18_cifar10_200epochs.pth  --eval-model resnet18 \
    --device cuda --epochs 500 --opt sgd --lr 0.2 --momentum 0.9 --weight-decay 1e-4 \
    --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --lr-warmup-decay 0.01 \
    --distill-data-path ./syn-data/cda_cifar10_rn18_1k_ipc2000 \
    -T 20 --image-per-class 250 --alpha 0.2 --curriculum-num 3 \
    --select-misclassified --select-method simple --balance \
    --score forgetting --score-path ./scores/forgetting_CIFAR10.npy \
    --output-dir ./selection_logs --num-eval 5
# ipc = 500
CUDA_VISIBLE_DEVICES=0, python /home/yanda/WORK/CCFS/ccfs_cifar10.py \
    --data-path /path/to/CIFAR-10/ --filter-model resnet18 --teacher-model resnet18 \
    --teacher-path ./checkpoints/resnet18_cifar10_200epochs.pth  --eval-model resnet18 \
    --device cuda --epochs 500 --opt sgd --lr 0.2 --momentum 0.9 --weight-decay 1e-4 \
    --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --lr-warmup-decay 0.01 \
    --distill-data-path ./syn-data/cda_cifar10_rn18_1k_ipc2000 \
    -T 20 --image-per-class 500 --alpha 0.1 --curriculum-num 3 \
    --select-misclassified --select-method simple --balance \
    --score forgetting --score-path ./scores/forgetting_CIFAR10.npy \
    --output-dir ./selection_logs --num-eval 5
# ipc = 1000
CUDA_VISIBLE_DEVICES=0, python /home/yanda/WORK/CCFS/ccfs_cifar10.py \
    --data-path /path/to/CIFAR-10/ --filter-model resnet18 --teacher-model resnet18 \
    --teacher-path ./checkpoints/resnet18_cifar10_200epochs.pth  --eval-model resnet18 \
    --device cuda --epochs 250 --opt sgd --lr 0.2 --momentum 0.9 --weight-decay 1e-4 \
    --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --lr-warmup-decay 0.01 \
    --distill-data-path ./syn-data/cda_cifar10_rn18_1k_ipc2000 \
    -T 20 --image-per-class 1000 --alpha 0.3 --curriculum-num 3 \
    --select-misclassified --select-method simple --balance \
    --score forgetting --score-path ./scores/forgetting_CIFAR10.npy \
    --output-dir ./selection_logs --num-eval 5
# ipc = 1500
CUDA_VISIBLE_DEVICES=0, python /home/yanda/WORK/CCFS/ccfs_cifar10.py \
    --data-path /path/to/CIFAR-10/ --filter-model resnet18 --teacher-model resnet18 \
    --teacher-path ./checkpoints/resnet18_cifar10_200epochs.pth  --eval-model resnet18 \
    --device cuda --epochs 200 --opt sgd --lr 0.2 --momentum 0.9 --weight-decay 1e-4 \
    --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --lr-warmup-decay 0.01 \
    --distill-data-path ./syn-data/cda_cifar10_rn18_1k_ipc2000 \
    -T 20 --image-per-class 1500 --alpha 0.3 --curriculum-num 3 \
    --select-misclassified --select-method simple --balance \
    --score forgetting --score-path ./scores/forgetting_CIFAR10.npy \
    --output-dir ./selection_logs --num-eval 5
