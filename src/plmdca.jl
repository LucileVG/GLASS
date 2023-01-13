using PlmDCA
using NPZ

path_msa = ARGS[1]
path_dca = ARGS[2]

dca_model = plmdca(path_msa, theta = 0.2, verbose = false)

npzwrite(path_dca, Dict("h" => dca_model.htensor, "J" => dca_model.Jtensor ))