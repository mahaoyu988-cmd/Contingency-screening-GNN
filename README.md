# Contingency-screening-GNN

initial_CS: This document is wring to run acopf and dcopf and based on initial dispatch run contingency screening by using acpf and dcpf

Dataset: This document is trying to create dataset by using different operational condition in a fixed outage sets

train_outage_lines = [36, 33, 8, 93, 94] (get from acpf)

val_outage_lines   = [36, 33, 8, 93, 94] 

test_outage_lines  = [38, 33, 142, 31, 93] (get from dcpf)


dynamic_edge_index: this code runs GNN by using dynamic edge_index and edge_attributes

fixed_edge_index: this code runs GNN by using fixed edge_index and edge_attributes
