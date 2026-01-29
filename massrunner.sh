for i in {1..10}; do
	python3 simulation.py --default --seed=$i
	python3 simulation.py --m=0 --seed=$i

	python3 table_exporter.py simulation_data_E30_F3_M3.json
	python3 table_exporter.py simulation_data_E30_F3_M0.json

	mv simulation_data_E30_F3_M3.csv M3_seed$i.csv
	mv simulation_data_E30_F3_M0.csv M0_seed$i.csv
done
