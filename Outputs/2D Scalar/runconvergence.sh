NCELL=$1
FILE=error.txt
rm -rf $FILE && touch $FILE
for ncell in $NCELL
do 
   echo "ncell = $ncell"
   python clw2d.py -Tf 1.0 -ncellx $ncell -ncelly $ncell -compute_error yes \
          -plot_freq 0 -scheme rk2 -limit mmod>log.txt
   tail -n 1 log.txt
   tail -n 1 log.txt >> $FILE
done
echo "Wrote file $FILE"