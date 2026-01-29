.PHONY: default

default:
	python3 simulation.py
	python3 visualize_graph.py simulation_data_E30_F3_M3.json
	open graph_E30_F3_M3.png

double:
	python3 simulation.py --default
	python3 simulation.py --m=0
	python3 visualize_graph.py simulation_data_E30_F3_M3.json simulation_data_E30_F3_M0.json
	open graph_E30_F3_M3_vs_E30_F3_M0.png

doublegraph:
	python3 visualize_graph.py simulation_data_E30_F3_M3.json simulation_data_E30_F3_M0.json
	open graph_E30_F3_M3_vs_E30_F3_M0.png

doublevideo:
	python3 visualization.py simulation_data_E30_F3_M3.json simulation_data_E30_F3_M0.json

withseed:
	python3 simulation.py --default --seed=3
	python3 simulation.py --m=0 --seed=3

gettables:
	python3 table_exporter.py simulation_data_E30_F3_M3.json
	python3 table_exporter.py simulation_data_E30_F3_M0.json
