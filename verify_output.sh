#!/usr/bin/bash
#
# Given an verifier output return 0 if everything passed

# https://stackoverflow.com/questions/5947742/how-to-change-the-output-color-of-echo-in-linux
RED='\033[0;31m'
NC='\033[0m' # No Color

# Bold
BBlack='\033[1;30m'       # Black
BRed='\033[1;31m'         # Red
BGreen='\033[1;32m'       # Green
BYellow='\033[1;33m'      # Yellow
BBlue='\033[1;34m'        # Blue
BPurple='\033[1;35m'      # Purple
BCyan='\033[1;36m'        # Cyan
BWhite='\033[1;37m'       # White

number_of_fails=$(grep FAIL $1|wc -l)
if [[ 0 -eq $number_of_fails  ]]
then
    echo -e "${BGreen}${1}: All Pass${NC}"
    exit 0
else
    echo -e "${BRed}${1}: Some failed${NC}"
    exit $number_of_fails
fi
