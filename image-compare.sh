result=$(compare $1 $2 -metric RMSE /dev/null 2>&1|grep -o "([0-9.]\+)"|grep -o "[0-9]\\.[0-9][0-9]"|sed "s/\.//"|sed "s/^0*//")
if [ $result == "" ]; then
  echo "0"
else
  echo $result
fi