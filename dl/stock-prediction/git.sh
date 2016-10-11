#!/bin/sh

git add $0
git commit -m $1
unset SSH_ASKPASS
git push origin master

