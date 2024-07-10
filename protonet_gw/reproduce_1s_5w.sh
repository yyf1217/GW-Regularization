# CUB 1s5w (convnet4)
python train_protonet_gw.py \
  --dataset CUB \
  --shot 1 \
  --max_epoch 1000 \
  --lr 0.001 \
  --step_size 100 \
  --gamma 0.8 \
  --c 0.01 \
  --model convnet \
  --hyperbolic \
  --not-riemannian \
  --dim 3200 \
  --h_dim 1600 \
  --num-layers 4 \
  --num-layers_post 2 \
  --post_optimizer RiemannianAdam \
  --beta 0.02


# CUB 1s5w (resnet10)
python train_protonet_gw.py \
  --dataset CUB \
  --shot 1 \
  --max_epoch 150 \
  --lr 0.001 \
  --step_size 100 \
  --gamma 0.8 \
  --c 0.01 \
  --model resnet10 \
  --hyperbolic \
  --not-riemannian \
  --dim 512 \
  --h_dim 512 \
  --num-layers 4 \
  --num-layers_post 2 \
  --post_optimizer RiemannianAdam \
  --beta 0.015

# CUB 1s5w (resnet12)
python train_protonet_gw.py \
  --dataset CUB \
  --shot 1 \
  --max_epoch 150 \
  --lr 0.001 \
  --step_size 100 \
  --gamma 0.8 \
  --c 0.01 \
  --model resnet12 \
  --hyperbolic \
  --not-riemannian \
  --dim 512 \
  --h_dim 512 \
  --num-layers 4 \
  --num-layers_post 2 \
  --post_optimizer RiemannianAdam \
  --beta 0.15

# CUB 1s5w (resnet18)
python train_protonet_gw.py \
  --dataset CUB \
  --shot 1 \
  --max_epoch 300 \
  --lr 0.001 \
  --step_size 100 \
  --gamma 0.8 \
  --c 0.01 \
  --model resnet12 \
  --hyperbolic \
  --not-riemannian \
  --dim 512 \
  --h_dim 512 \
  --num-layers 4 \
  --num-layers_post 2 \
  --post_optimizer RiemannianAdam \
  --beta 0.1

# MiniImageNet 1s5w (convnet4)
python train_protonet_gw.py \
  --dataset MiniImageNet \
  --shot 1 \
  --max_epoch 500 \
  --lr 0.0005 \
  --step_size 300 \
  --gamma 0.8 \
  --c 0.08 \
  --model convnet \
  --hyperbolic \
  --not-riemannian \
  --dim 3200 \
  --h_dim 1600 \
  --num-layers 4 \
  --num-layers_post 3 \
  --post_optimizer RiemannianAdam \
  --beta 0.6


# MiniImageNet 1s5w (resnet10)
python train_protonet_gw.py \
  --dataset MiniImageNet \
  --shot 1 \
  --max_epoch 500 \
  --lr 0.001 \
  --step_size 200 \
  --gamma 0.8 \
  --c 0.08 \
  --model resnet10 \
  --hyperbolic \
  --not-riemannian \
  --dim 512 \
  --h_dim 512 \
  --num-layers 4 \
  --num-layers_post 3 \
  --post_optimizer RiemannianAdam \
  --beta 0.5

# MiniImageNet 1s5w (resnet12)
python train_protonet_gw.py \
  --dataset MiniImageNet \
  --shot 1 \
  --max_epoch 500 \
  --lr 0.0005 \
  --step_size 200 \
  --gamma 0.8 \
  --c 0.08 \
  --model resnet10 \
  --hyperbolic \
  --not-riemannian \
  --dim 512 \
  --h_dim 512 \
  --num-layers 4 \
  --num-layers_post 3 \
  --post_optimizer RiemannianAdam \
  --beta 0.1

# MiniImageNet 1s5w (resnet12)
python train_protonet_gw.py \
  --dataset MiniImageNet \
  --shot 1 \
  --max_epoch 500 \
  --lr 0.0005 \
  --step_size 200 \
  --gamma 0.8 \
  --c 0.08 \
  --model resnet10 \
  --hyperbolic \
  --not-riemannian \
  --dim 512 \
  --h_dim 512 \
  --num-layers 4 \
  --num-layers_post 3 \
  --post_optimizer RiemannianAdam \
  --beta 0.7