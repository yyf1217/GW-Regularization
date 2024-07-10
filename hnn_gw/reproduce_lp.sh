# dataset disease

python train.py --task lp --dataset disease_lp --model HGCN --lr 0.01 --dim 3 --num-layers 2 --beta 0.3 --act relu --bias 1 --dropout 0.0 --weight-decay 0.0 --manifold PoincareBall

python train.py --task lp --dataset disease_lp --model HNN --lr 0.01 --dim 3 --num-layers 2 --beta 0.9 --act relu --bias 1 --dropout 0.0 --weight-decay 0.0 --manifold PoincareBall

python train.py --task lp --dataset disease_lp --model HyboNet --lr 0.005 --dim 16 --num-layers 2 --beta 0.3 --act relu --bias 1 --dropout 0.0 --weight-decay 0.0 --manifold Lorentz



# dataset airport

python train.py --task lp --dataset airport --model HGCN --lr 0.01 --dim 16 --num-layers 2 --beta 2 --act relu --bias 1 --dropout 0.0 --weight-decay 0.0005 --manifold PoincareBall

python train.py --task lp --dataset airport --model HNN --lr 0.01 --dim 16 --num-layers 2 --beta 2 --act relu --bias 1 --dropout 0.0 --weight-decay 0.0005 --manifold PoincareBall

python train.py --task lp --dataset airport --model HyboNet --lr 0.01 --dim 16 --num-layers 2 --beta 1.5 --act relu --bias 1 --dropout 0.0 --weight-decay 0.0 --manifold Lorentz



# dataset cora

python train.py --task lp --dataset cora --model HGCN --lr 0.01 --dim 16 --num-layers 2 --beta 0.1 --act relu --bias 1 --dropout 0.5 --weight-decay 0.005 --manifold PoincareBall

python train.py --task lp --dataset cora --model HNN --lr 0.01 --dim 16 --num-layers 2 --beta 0.5 --act relu --bias 1 --dropout 0.5 --weight-decay 0.005 --manifold PoincareBall

python train.py --task lp --dataset cora --model HyboNet --lr 0.02 --dim 16 --num-layers 2 --beta 0.25 --act relu --bias 1 --dropout 0.7 --weight-decay 0.001 --manifold Lorentz