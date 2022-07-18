#!/bin/sh

git filter-branch --env-filter '

OLD_EMAIL="gmchaudhari6@gmail.com"
CORRECT_NAME="Gaurav Chaudhari"
CORRECT_EMAIL="gauravbyte@gmail.com"

if [ "$GIT_COMMITTER_EMAIL" = "$gmchaudhari6@gmail.com" ]
then
    export GIT_COMMITTER_NAME="$Gaurav Chaudhari"
    export GIT_COMMITTER_EMAIL="$gauravbyte@gmail.com"
fi
if [ "$GIT_AUTHOR_EMAIL" = "$gmchaudhari6@gmail.com	" ]
then
    export GIT_AUTHOR_NAME="$Gaurav Chaudhari"
    export GIT_AUTHOR_EMAIL="$gauravbyte@gmail.com"
fi
' --tag-name-filter cat -- --branches --tags
