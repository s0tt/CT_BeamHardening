#pragma once
#include <torch/script.h>
#include <torch/torch.h>
#include <VoxieClient/Array.hpp>
#include <VoxieClient/ClaimedOperation.hpp>
#include <VoxieClient/DBusTypeList.hpp>

#include <array>
#include <cmath>
#include <iostream>

#include <QObject>

class CNNAiCt {
 public:
  CNNAiCt(torch::jit::script::Module module);
  void infere(vx::Array3<const float>& inputVolume,
              vx::Array3<float>& outputVolume, int batchSize,
              vx::ClaimedOperation<
                  de::uni_stuttgart::Voxie::ExternalOperationRunFilter>& prog);

 private:
  torch::jit::script::Module module;
};
