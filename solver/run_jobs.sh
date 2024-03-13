#!/bin/bash

for Mach in $(seq 0.1 0.01 0.25);
do
	for BackPres in 97250.0 97000.0 97500.0
	do
		folder=Mach_${Mach}_BackPres_${BackPres}
		mkdir -p ${folder}
		cp lam_flatplatecomp.cfg ${folder}/config.cfg
		cp mesh_flatplate_65x65.su2 ${folder}/mesh_flatplate_65x65.su2
		sed -i "s/MACH_NUMBER= 0.2/MACH_NUMBER= ${Mach}/g" ${folder}/config.cfg
		sed -i "s/MARKER_OUTLET= ( outlet, 97250.0, farfield, 97250.0 )/MARKER_OUTLET= ( outlet, ${BackPres}, farfield, ${BackPres} )/g" ${folder}/config.cfg
		sed -i "s/VOLUME_FILENAME= flow_comp/VOLUME_FILENAME= flow_comp_${Mach}_${BackPres}/g" ${folder}/config.cfg
		cd ${folder}
		mpirun -n 4 SU2_CFD config.cfg
		cd -
	done
done

