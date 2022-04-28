#!/bin/bash
files=(`ls -v ${1}/*${2}`)
len=${#files[@]}
# for i in "${files[@]}";do echo $i;done
for (( i=0; i<$len; i++ ));do 
	mv ${files[$i]} ${1}/$((${i}+1))${2}.tmp;
	echo "mv ${files[$i]} ${1}/$((${i}+1))${2}";
done
for (( i=0; i<$len; i++ ));do 
	mv ${1}/$((${i}+1))${2}.tmp ${1}/$((${i}+1))${2};
done
