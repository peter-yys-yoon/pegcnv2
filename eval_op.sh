config=$1
tag=$1$2
aa="python main.py -e -c ${config} -ot 10 -t ${tag}"
#echo ${aa}
#python main.py -e -c ${config} -op 1 -t ${tag} -bs 128
python main.py -e -c ${config} -op 2 -t ${tag} -bs 128
python main.py -e -c ${config} -op 3 -t ${tag} -bs 128
python main.py -e -c ${config} -op 4 -t ${tag} -bs 128
python main.py -e -c ${config} -op 5 -t ${tag} -bs 128
