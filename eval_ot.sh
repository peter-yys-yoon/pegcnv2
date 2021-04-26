config=$1
tag=$1$2
aa="python main.py -e -c ${config} -ot 10 -t ${tag}"
python main.py -e -c ${config} -ot 10 -t ${tag} -bs 128
python main.py -e -c ${config} -ot 20 -t ${tag} -bs 128
python main.py -e -c ${config} -ot 30 -t ${tag} -bs 128
python main.py -e -c ${config} -ot 40 -t ${tag} -bs 128
python main.py -e -c ${config} -ot 50 -t ${tag} -bs 128
#echo ${aa}

