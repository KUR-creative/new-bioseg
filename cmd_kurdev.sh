#python train.py ./experiments/b_32_4.yml
#python train.py ./experiments/b_32_5.yml
#python train.py ./experiments/b_64_4.yml
#python train.py ./experiments/b_64_5.yml
while true
do
    python train.py ./experiments/jacc_kurdev/m_32_4.yml
    python train.py ./experiments/jacc_kurdev/m_32_5.yml
    python train.py ./experiments/jacc_kurdev/m_64_4.yml
    python train.py ./experiments/jacc_kurdev/m_64_5.yml
    python train.py ./experiments/jacc_kurdev/bm_32_5.yml
    python train.py ./experiments/jacc_kurdev/bm_64_5.yml
done
