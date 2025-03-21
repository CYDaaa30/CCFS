# ipc = 25
CUDA_VISIBLE_DEVICES=0, python /home/yanda/WORK/CCFS/ccfs_cifar100.py \
    --data-path /path/to/CIFAR-100/ --filter-model resnet18 --teacher-model resnet18 \
    --teacher-path ./checkpoints/resnet18_cifar100_200epochs.pth  --eval-model resnet18 \
    --device cuda --epochs 500 --opt sgd --lr 0.2 --momentum 0.9 --weight-decay 1e-4 \
    --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --lr-warmup-decay 0.01 \
    --distill-data-path ./syn-data/cda_cifar100_rn18_1k_ipc200 \
    -T 20 --image-per-class 25 --alpha 0.2 --curriculum-num 3 \
    --select-misclassified --select-method simple --balance \
    --score forgetting --score-path ./scores/forgetting_CIFAR100.npy \
    --output-dir ./selection_logs --num-eval 5
# ipc = 50
CUDA_VISIBLE_DEVICES=0, python /home/yanda/WORK/CCFS/ccfs_cifar100.py \
    --data-path /path/to/CIFAR-100/ --filter-model resnet18 --teacher-model resnet18 \
    --teacher-path ./checkpoints/resnet18_cifar100_200epochs.pth  --eval-model resnet18 \
    --device cuda --epochs 500 --opt sgd --lr 0.2 --momentum 0.9 --weight-decay 1e-4 \
    --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --lr-warmup-decay 0.01 \
    --distill-data-path ./syn-data/cda_cifar100_rn18_1k_ipc200 \
    -T 20 --image-per-class 50 --alpha 0.1 --curriculum-num 3 \
    --select-misclassified --select-method simple --balance \
    --score forgetting --score-path ./scores/forgetting_CIFAR100.npy \
    --output-dir ./selection_logs --num-eval 5
# ipc = 100
CUDA_VISIBLE_DEVICES=0, python /home/yanda/WORK/CCFS/ccfs_cifar100.py \
    --data-path /path/to/CIFAR-100/ --filter-model resnet18 --teacher-model resnet18 \
    --teacher-path ./checkpoints/resnet18_cifar100_200epochs.pth  --eval-model resnet18 \
    --device cuda --epochs 250 --opt sgd --lr 0.2 --momentum 0.9 --weight-decay 1e-4 \
    --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --lr-warmup-decay 0.01 \
    --distill-data-path ./syn-data/cda_cifar100_rn18_1k_ipc200 \
    -T 20 --image-per-class 100 --alpha 0.1 --curriculum-num 3 \
    --select-misclassified --select-method simple --balance \
    --score forgetting --score-path ./scores/forgetting_CIFAR100.npy \
    --output-dir ./selection_logs --num-eval 5
# ipc = 150
CUDA_VISIBLE_DEVICES=0, python /home/yanda/WORK/CCFS/ccfs_cifar100.py \
    --data-path /path/to/CIFAR-100/ --filter-model resnet18 --teacher-model resnet18 \
    --teacher-path ./checkpoints/resnet18_cifar100_200epochs.pth  --eval-model resnet18 \
    --device cuda --epochs 200 --opt sgd --lr 0.2 --momentum 0.9 --weight-decay 1e-4 \
    --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --lr-warmup-decay 0.01 \
    --distill-data-path ./syn-data/cda_cifar100_rn18_1k_ipc200 \
    -T 20 --image-per-class 150 --alpha 0.2 --curriculum-num 3 \
    --select-misclassified --select-method simple --balance \
    --score forgetting --score-path ./scores/forgetting_CIFAR100.npy \
    --output-dir ./selection_logs --num-eval 5
