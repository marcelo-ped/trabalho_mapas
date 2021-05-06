#!/bin/bash
rm *.dat
cp temain_mod.f.txt /tmp/temain_mod.f
cp teprob.f.txt /tmp/teprob.f
sed -i "s-~-.-g" /tmp/temain_mod.f
random_num_old=$(sed '1187!d' teprob.f.txt | tr -dc '0-9')
gfortran /tmp/temain_mod.f /tmp/teprob.f -o temain_mod_original
./temain_mod_original
mv *.dat fault_1

sed -i "s/IDV(1)=1/IDV(2)=1/g" temain_mod.f.txt
cp temain_mod.f.txt /tmp/temain_mod.f
random_num_new=$(($RANDOM%1000000000))
sed -i "s/G=${random_num_old}.D0/G=${random_num_new}.D0/g" teprob.f.txt
cp teprob.f.txt /tmp/teprob.f
sed -i "s-~-.-g" /tmp/temain_mod.f
gfortran /tmp/temain_mod.f /tmp/teprob.f -o temain_mod_original
./temain_mod_original
mv *.dat fault_1
rm *.dat
sed -i "s/IDV(2)=1/IDV(1)=1/g" temain_mod.f.txt
