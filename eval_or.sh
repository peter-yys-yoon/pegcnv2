config=$1
tag=$1$2
aa="python main.py -e -c ${config} -ot 10 -t ${tag}"
python main.py -e -c ${config} -or 0.2 -t ${tag} -bs 128
python main.py -e -c ${config} -or 0.3 -t ${tag} -bs 128
python main.py -e -c ${config} -or 0.4 -t ${tag} -bs 128
python main.py -e -c ${config} -or 0.5 -t ${tag} -bs 128
python main.py -e -c ${config} -or 0.6 -t ${tag} -bs 128

