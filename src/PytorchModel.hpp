#pragma once
#include <torch/torch.h>
#include <torch/script.h>
#include <VoxieClient/Array.hpp>
#include <VoxieClient/ClaimedOperation.hpp>
#include <VoxieClient/DBusTypeList.hpp>


#include <array>
#include <cmath>
#include <iostream>

#include <QObject>

class PytorchModel {
 public:
  PytorchModel();
  void infere(vx::Array3<const float>& inputVolume,
               vx::Array3<float>& outputVolume,
               vx::ClaimedOperation<
                   de::uni_stuttgart::Voxie::ExternalOperationRunFilter>& prog);
};
