set title "loss value vs epochs"
set xlabel "epochs"
set ylabel "loss value"
plot "output.txt" with lines title "loss value curve"

pause -1 "Hit any key to continue"
