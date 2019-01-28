import 'dart:typed_data';

import 'package:ml_algo/src/classifier/labels_processor/labels_processor.dart';
import 'package:ml_linalg/vector.dart';

class LabelsProcessorImpl<T> implements LabelsProcessor<T> {
  final _float32x4Zeroes = Float32x4.zero();
  final _float32x4Ones = Float32x4.splat(1.0);

  @override
  MLVector<T> makeLabelsOneVsAll(MLVector<T> origLabels, double targetLabel) {
    switch (T) {
      case Float32x4:
        return _makeFloat32x4LabelsOneVsAll(origLabels as MLVector<Float32x4>, targetLabel) as MLVector<T>;
      default:
        throw UnimplementedError();
    }
  }

  MLVector<Float32x4> _makeFloat32x4LabelsOneVsAll(MLVector<Float32x4> origLabels, double targetLabel) {
    final targetAsFloat32x4 = Float32x4.splat(targetLabel);
    return origLabels
        .fastMap((Float32x4 element, int start, int end) => element.equal(targetAsFloat32x4)
        .select(_float32x4Ones, _float32x4Zeroes));
  }
}
