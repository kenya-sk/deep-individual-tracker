FILE="video2image"
if [ -e $FILE ];then
	echo "remove $FILE (old version)"
	rm $FILE
fi

make
