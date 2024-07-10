# dataset disease

python train.py --task nc --dataset disease_nc --model HGCN --lr 0.01 --dim 16 --num-layers 2 --beta 0.09 --act relu --bias 1 --dropout 0.1 --weight-decay 0.001 --manifold PoincareBall

python train.py --task nc --dataset disease_nc --model HNN --lr 0.01 --dim 16 --num-layers 2 --beta 0.08 --act relu --bias 1 --dropout 0.1 --weight-decay 0.001 --manifold PoincareBall

python train.py --task nc --dataset disease_nc --model HyboNet --lr 0.005 --dim 16 --num-layers 4 --beta 0.07 --act relu --bias 1 --dropout 0.1 --weight-decay 0.0 --manifold Lorentz



# dataset airport

python train.py --task nc --dataset airport --model HGCN --lr 0.01 --dim 16 --num-layers 2 --beta 1.5 --act relu --bias 1 --dropout 0.0 --weight-decay 0.0 --manifold PoincareBall

python train.py --task nc --dataset airport --model HNN --lr 0.01 --dim 16 --num-layers 2 --beta 1.25 --act relu --bias 1 --dropout 0.0 --weight-decay 0.0 --manifold PoincareBall

python train.py --task nc --dataset airport --model HyboNet --lr 0.02 --dim 16 --num-layers 6 --beta 1.1 --act relu --bias 1 --dropout 0.0001 --weight-decay 0.0 --manifold Lorentz



# dataset cora

python train.py --task nc --dataset cora --model HGCN --lr 0.01 --dim 16 --num-layers 2 --beta 0.6 --act relu --bias 1 --dropout 0.5 --weight-decay 0.001 --manifold PoincareBall

python train.py --task nc --dataset cora --model HNN --lr 0.01 --dim 16 --num-layers 2 --beta 0.35 --act relu --bias 1 --dropout 0.5 --weight-decay 0.001 --manifold PoincareBall

python train.py --task nc --dataset cora --model HyboNet --lr 0.02 --dim 16 --num-layers 3 --beta 0.25 --act relu --bias 1 --dropout 0.7 --weight-decay 0.01 --manifold Lorentz



# dataset cornell

python train.py --task nc --dataset cornell --model HGCN --lr 0.01 --dim 16 --num-layers 2 --beta 0.25 --act relu --bias 1 --dropout 0.5 --weight-decay 0.001 --manifold PoincareBall

python train.py --task nc --dataset cornell --model HNN --lr 0.01 --dim 16 --num-layers 2 --beta 0.25 --act relu --bias 1 --dropout 0.5 --weight-decay 0.001 --manifold PoincareBall

python train.py --task nc --dataset cornell --model HyboNet --lr 0.005 --dim 16 --num-layers 2 --beta 0.13 --act relu --bias 1 --dropout 0.2 --weight-decay 0.0 --manifold Lorentz



# dataset texas

python train.py --task nc --dataset texas --model HGCN --lr 0.01 --dim 16 --num-layers 2 --beta 0.15 --act relu --bias 1 --dropout 0.5 --weight-decay 0.001 --manifold PoincareBall

python train.py --task nc --dataset texas --model HNN --lr 0.01 --dim 16 --num-layers 2 --beta 0.1 --act relu --bias 1 --dropout 0.5 --weight-decay 0.001 --manifold PoincareBall

python train.py --task nc --dataset texas --model HyboNet --lr 0.005 --dim 16 --num-layers 2 --beta 0.1 --act relu --bias 1 --dropout 0.2 --weight-decay 0.0 --manifold Lorentz



# dataset wisconsin

python train.py --task nc --dataset wisconsin --model HGCN --lr 0.01 --dim 16 --num-layers 2 --beta 0.2 --act relu --bias 1 --dropout 0.5 --weight-decay 0.001 --manifold PoincareBall

python train.py --task nc --dataset wisconsin --model HNN --lr 0.01 --dim 16 --num-layers 2 --beta 0.15 --act relu --bias 1 --dropout 0.5 --weight-decay 0.001 --manifold PoincareBall

python train.py --task nc --dataset wisconsin --model HyboNet --lr 0.005 --dim 16 --num-layers 2 --beta 0.1 --act relu --bias 1 --dropout 0.2 --weight-decay 0.0 --manifold Lorentz



# dataset chiameleon

python train.py --task nc --dataset chiameleon --model HGCN --lr 0.01 --dim 16 --num-layers 2 --beta 0.08 --act relu --bias 1 --dropout 0.5 --weight-decay 0.001 --manifold PoincareBall

python train.py --task nc --dataset chiameleon --model HNN --lr 0.01 --dim 16 --num-layers 2 --beta 0.02 --act relu --bias 1 --dropout 0.5 --weight-decay 0.001 --manifold PoincareBall

python train.py --task nc --dataset chiameleon --model HyboNet --lr 0.005 --dim 16 --num-layers 2 --beta 1.3 --act relu --bias 1 --dropout 0.2 --weight-decay 0.0 --manifold Lorentz



# dataset squirrel

python train.py --task nc --dataset squirrel --model HGCN --lr 0.01 --dim 5 --num-layers 2 --beta 0.01 --act relu --bias 1 --dropout 0.5 --weight-decay 0.001 --manifold PoincareBall

python train.py --task nc --dataset squirrel --model HNN --lr 0.01 --dim 5 --num-layers 2 --beta 0.001 --act relu --bias 1 --dropout 0.5 --weight-decay 0.001 --manifold PoincareBall

python train.py --task nc --dataset squirrel --model HyboNet --lr 0.005 --dim 16 --num-layers 2 --beta 2.5 --act relu --bias 1 --dropout 0.2 --weight-decay 0.0 --manifold Lorentz



# dataset actor

python train.py --task nc --dataset film --model HGCN --lr 0.01 --dim 5 --num-layers 2 --beta 0.03 --act relu --bias 1 --dropout 0.5 --weight-decay 0.001 --manifold PoincareBall

python train.py --task nc --dataset film --model HNN --lr 0.01 --dim 5 --num-layers 2 --beta 0.03 --act relu --bias 1 --dropout 0.5 --weight-decay 0.001 --manifold PoincareBall

python train.py --task nc --dataset film --model HyboNet --lr 0.005 --dim 16 --num-layers 2 --beta 0.01 --act relu --bias 1 --dropout 0.2 --weight-decay 0.0 --manifold Lorentz