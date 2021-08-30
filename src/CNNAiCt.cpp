#include "CNNAiCt.hpp"

CNNAiCt::CNNAiCt(torch::jit::script::Module module) : module(module) {}

void CNNAiCt::infere(
    vx::Array3<float const>& inputVolume, vx::Array3<float>& outputVolume,
    int batchSize,
    vx::ClaimedOperation<de::uni_stuttgart::Voxie::ExternalOperationRunFilter>&
        prog) {
  int nx = inputVolume.size<0>();
  int ny = inputVolume.size<1>();
  int nz = inputVolume.size<2>();

  auto data = inputVolume.data();

  // TODO remove const_cast and clone data instead?
  torch::Tensor inputTensor =
      torch::from_blob(const_cast<float*>(data), {nz, ny, nx});

  // transpose z and x to get back actual dimensions
  inputTensor = inputTensor.transpose(0, 2);

  std::vector<torch::Tensor> batchList;
  std::vector<int> indices;
  torch::TensorList tensors;

  // iterate over volume in y direction
  for (int y = 0; y < ny; y++) {
    torch::Tensor sample;
    if (y < 2) {
      // Handle lower borders by fill missing lower neighbours with first slice
      auto sliceFirst =
          inputTensor
              .index({torch::indexing::Slice(), torch::indexing::Slice(0, 1),
                      torch::indexing::Slice()})
              .transpose(0, 1);

      for (int i = 0; i < (2 - y); i++) {
        i == 0 ? sample = sliceFirst
               : sample = torch::cat({sample, sliceFirst});
      }

      sample = torch::cat({sample, inputTensor
                                       .index({torch::indexing::Slice(),
                                               torch::indexing::Slice(0, 3 + y),
                                               torch::indexing::Slice()})
                                       .transpose(0, 1)});
      sample = sample.unsqueeze(0);
    } else if (y > (ny - 3)) {
      // Handle upper borders by fill missing upper neighbours with last slice
      sample = inputTensor
                   .index({torch::indexing::Slice(),
                           torch::indexing::Slice(y - 2, ny),
                           torch::indexing::Slice()})
                   .transpose(0, 1);
      auto sliceLast = inputTensor
                           .index({torch::indexing::Slice(),
                                   torch::indexing::Slice(ny - 1, ny),
                                   torch::indexing::Slice()})
                           .transpose(0, 1);
      for (int i = 0; i < (y - (ny - 3)); i++) {
        sample = torch::cat({sample, sliceLast});
      }
      sample = sample.unsqueeze(0);

    } else {
      sample = inputTensor
                   .index({torch::indexing::Slice(),
                           torch::indexing::Slice(y - 2, y + 3),
                           torch::indexing::Slice()})
                   .transpose(0, 1)
                   .unsqueeze(0);
    }

    batchList.push_back(sample);
    indices.push_back(y);

    // check if enough samples for specified batch size or last batch
    if (batchList.size() == batchSize || y == ny - 1) {
      // cat samples to 4-dim tensor with (sample_dim, slice_dim, y_dim, z_dim)
      auto batch = torch::cat({batchList});

      std::vector<torch::jit::IValue> inputs;
      inputs.push_back(batch);
      at::Tensor outputTensor = module.forward(inputs).toTensor();

      // write output tensor to voxie volume
      int outputIdx = 0;
      for (int inputIdx : indices) {
        for (int x = 0; x < nx; x++) {
          for (int z = 0; z < nz; z++) {
            outputVolume(x, inputIdx, z) =
                outputTensor[outputIdx][0][x][z].item<float>();
          }
        }
        outputIdx++;
      }
      indices.clear();
      inputs.clear();
      batchList.clear();
    }
    HANDLEDBUSPENDINGREPLY(
        prog.opGen().SetProgress(((float)y) / ny, vx::emptyOptions()));
  }

  HANDLEDBUSPENDINGREPLY(prog.opGen().SetProgress(1.00, vx::emptyOptions()));
}
