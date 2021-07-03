FILE="video2img"
if [ -e $FILE ];then
	echo "remove $FILE (old version)"
	rm $FILE
fi

make
