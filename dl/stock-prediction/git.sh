#!/bin/sh

git add $0
git commit -m $1
unset SSH_ASKPASS
if [ "$?"="0" ];then
	git push origin +master
else
	git push origin master
fi
