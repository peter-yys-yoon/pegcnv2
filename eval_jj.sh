config=$1
tag=$1$2
aa="python main.py -e -c ${config} -ot 10 -t ${tag}"
python main.py -e -c ${config} -jj 0.02 -sigma 0.1 -t ${tag} -bs 128
python main.py -e -c ${config} -jj 0.04 -sigma 0.1 -t ${tag} -bs 128
python main.py -e -c ${config} -jj 0.06 -sigma 0.1 -t ${tag} -bs 128
python main.py -e -c ${config} -jj 0.08 -sigma 0.1 -t ${tag} -bs 128
python main.py -e -c ${config} -jj 0.1 -sigma 0.1 -t ${tag} -bs 128

