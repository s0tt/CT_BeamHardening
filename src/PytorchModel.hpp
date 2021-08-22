#ifndef LOCALTHICKNESS_H
#define LOCALTHICKNESS_H

#include <VoxieClient/Array.hpp>
#include <VoxieClient/ClaimedOperation.hpp>
#include <VoxieClient/DBusTypeList.hpp>

#include <array>
#include <cmath>
#include <iostream>

#include <QObject>

class LocalThickness {
 public:
  LocalThickness();
  void compute(vx::Array3<const float>& inputVolume,
               vx::Array3<float>& outputVolume,
               vx::ClaimedOperation<
                   de::uni_stuttgart::Voxie::ExternalOperationRunFilter>& prog);
  void computeNaive(
      vx::Array3<const float>& inputVolume, vx::Array3<float>& outputVolume,
      vx::ClaimedOperation<
          de::uni_stuttgart::Voxie::ExternalOperationRunFilter>& prog);
};

#endif  // LOCALTHICKNESS_H
