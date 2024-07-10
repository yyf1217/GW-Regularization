# CUB 5s5w (convnet4)
python train_protonet_gw.py \
  --dataset CUB \
  --shot 5 \
  --max_epoch 800 \
  --lr 0.001 \
  --step_size 100 \
  --gamma 0.8 \
  --c 0.01 \
  --model convnet \
  --hyperbolic \
  --not-riemannian \
  --dim 4000 \
  --h_dim 1600 \
  --num-layers 4 \
  --num-layers_post 3 \
  --post_optimizer RiemannianAdam \
  --beta 0.07

# CUB 5s5w (resnet10)
python train_protonet_gw.py \
  --dataset CUB \
  --shot 5 \
  --max_epoch 300 \
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
  --num-layers_post 3 \
  --post_optimizer RiemannianAdam \
  --beta 0.01

# CUB 5s5w (resnet12)
python train_protonet_gw.py \
  --dataset CUB \
  --shot 5 \
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
  --num-layers_post 3 \
  --post_optimizer RiemannianAdam \
  --beta 0.25


# CUB 5s5w (resnet18)
python train_protonet_gw.py \
  --dataset CUB \
  --shot 5 \
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
  --num-layers_post 3 \
  --post_optimizer RiemannianAdam \
  --beta 0.1

# MiniImageNet 5s5w (convnet4)
python train_protonet_gw.py \
  --dataset MiniImageNet \
  --shot 5 \
  --max_epoch 500 \
  --lr 0.01 \
  --step_size 60 \
  --gamma 0.8 \
  --c 0.001 \
  --model convnet \
  --hyperbolic \
  --not-riemannian \
  --dim 4000 \
  --h_dim 1600 \
  --num-layers 4 \
  --num-layers_post 3 \
  --post_optimizer RiemannianAdam \
  --beta 14

# MiniImageNet 5s5w (resnet10)
python train_protonet_gw.py \
  --dataset MiniImageNet \
  --shot 5 \
  --max_epoch 300 \
  --lr 0.001 \
  --step_size 60 \
  --gamma 0.8 \
  --c 0.001 \
  --model resnet10 \
  --hyperbolic \
  --not-riemannian \
  --dim 512 \
  --h_dim 64 \
  --num-layers 4 \
  --num-layers_post 3 \
  --post_optimizer RiemannianAdam \
  --beta 0.1

# MiniImageNet 5s5w (resnet12)
python train_protonet_gw.py \
  --dataset MiniImageNet \
  --shot 5 \
  --max_epoch 300 \
  --lr 0.001 \
  --step_size 60 \
  --gamma 0.8 \
  --c 0.001 \
  --model resnet12 \
  --hyperbolic \
  --not-riemannian \
  --dim 512 \
  --h_dim 64 \
  --num-layers 4 \
  --num-layers_post 3 \
  --post_optimizer RiemannianAdam \
  --beta 0.5

# MiniImageNet 5s5w (resnet18)
python train_protonet_gw.py \
  --dataset MiniImageNet \
  --shot 5 \
  --max_epoch 300 \
  --lr 0.001 \
  --step_size 60 \
  --gamma 0.8 \
  --c 0.001 \
  --model resnet18 \
  --hyperbolic \
  --not-riemannian \
  --dim 512 \
  --h_dim 64 \
  --num-layers 4 \
  --num-layers_post 3 \
  --post_optimizer RiemannianAdam \
  --beta 0.5