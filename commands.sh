#!/bin/bash
cp temain_mod.f.txt /tmp/temain_mod.f
cp teprob.f.txt /tmp/teprob.f
sed -i "s-~-.-g" /tmp/temain_mod.f
random_num_old=$(sed '1187!d' teprob.f.txt | tr -dc '0-9')
gfortran /tmp/temain_mod.f /tmp/teprob.f -o temain_mod_original
./temain_mod_original
cp TE_data_me01.dat /home/marcelo/Desktop/TE_data_me01_0.dat
rm *.dat
for num_file in $(seq 1 99)
do
	cp temain_mod.f.txt /tmp/temain_mod.f
	random_num_new=$(($RANDOM%1000000000))
	sed -i "s/G=${random_num_old}.D0/G=${random_num_new}.D0/g" teprob.f.txt
	cp teprob.f.txt /tmp/teprob.f
	sed -i "s-~-.-g" /tmp/temain_mod.f
	gfortran /tmp/temain_mod.f /tmp/teprob.f -o temain_mod_original
	./temain_mod_original
	cp TE_data_me01.dat /home/marcelo/Desktop/TE_data_me01_${num_file}.dat
	rm *.dat
	random_num_old=$random_num_new
done
