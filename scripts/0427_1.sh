# python utils/04_build_embeddings_from_global_meta.py protein \
#   --global-meta data/training_ready/ptv3/global_meta.json \
#   --output-pkl data/training_ready/ptv3/derived/protein_embedding_esm.pkl \
#   --fasta /root/tmp/proteintalk_v2/data/training_ready/ptv3/derived/idmapping_2026_04_27.fasta \
#   --model-name /root/beam_wuhao/hf_cache/models--facebook--esm2_t33_650M_UR50D/snapshots/08e4846e537177426273712802403f7ba8261b6c \
#   --batch-size 4 \
#   --max-length 1024

# python utils/04_build_embeddings_from_global_meta.py drug \
#   --global-meta data/training_ready/ptv3/global_meta.json \
#   --output-pkl data/training_ready/ptv3/derived/drug_embedding_morgan_2048.pkl \
#   --radius 2 \
#   --n-bits 2048

python utils/05_build_graph_matrices_from_global_meta.py ppi \
  --global-meta data/training_ready/ptv3/global_meta.json \
  --edge-path '/root/beam_wuhao/H100/vcc_data/westlake/20250410_6508308PPI_protein_links_detailed_v12_both_prot1&2_.csv' \
  --output-npy data/training_ready/ptv3/derived/ppi_matrix.npy


# python utils/05_build_graph_matrices_from_global_meta.py ddi \
#   --global-meta data/training_ready/ptv3/global_meta.json \
#   --output-npy data/training_ready/ptv3/derived/ddi_matrix.npy \
#   --radius 2 \
#   --n-bits 2048


# python utils/05_build_graph_matrices_from_global_meta.py pdi \
#   --global-meta data/training_ready/ptv3/global_meta.json \
#   --output-npy data/training_ready/ptv3/derived/pdi_matrix.npy \
#   --stitch-db-dir /mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/stitch_db


python utils/08_visualize_graph_matrix_distributions.py \
  --derived-dir data/training_ready/ptv3/derived \
  --output-dir data/training_ready/ptv3/derived/graph_value_distributions