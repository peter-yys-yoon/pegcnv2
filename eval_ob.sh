config=$1
tag=$1$2
aa="python main.py -e -c ${config} -ot 10 -t ${tag}"
python main.py -e -c ${config} -ob 1 -t ${tag} -bs 128
python main.py -e -c ${config} -ob 2 -t ${tag} -bs 128
python main.py -e -c ${config} -ob 3 -t ${tag} -bs 128
python main.py -e -c ${config} -ob 4 -t ${tag} -bs 128
python main.py -e -c ${config} -ob 5 -t ${tag} -bs 128
#echo ${aa}

