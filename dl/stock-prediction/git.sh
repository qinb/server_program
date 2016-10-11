#!/bin/sh

git add $0
git commit -m $1
SSH_ACKPASS
git push origin master

