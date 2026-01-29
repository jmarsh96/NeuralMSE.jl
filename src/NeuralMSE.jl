module NeuralMSE

using BSON
import BSON: load

using NeuralEstimators
import NeuralEstimators: sampleposterior

# Public API
export infer_nbe, infer_npe, prepare_data

include("functions.jl")

end
