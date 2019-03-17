#python train.py ./experiments/b_32_4.yml
#python train.py ./experiments/b_32_5.yml
#python train.py ./experiments/b_64_4.yml
#python train.py ./experiments/b_64_5.yml
while true
do
    python train.py ./experiments/jacc_server/m_32_4.yml
    python train.py ./experiments/jacc_server/m_32_5.yml
    python train.py ./experiments/jacc_server/m_64_4.yml
    python train.py ./experiments/jacc_server/m_64_5.yml
    python train.py ./experiments/jacc_server/bm_32_5.yml
    python train.py ./experiments/jacc_server/bm_64_5.yml
done
