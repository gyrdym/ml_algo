import 'dart:typed_data';

import 'package:ml_algo/src/classifier/labels_processor/labels_processor.dart';
import 'package:ml_linalg/float32x4_vector.dart';
import 'package:ml_linalg/vector.dart';

class LabelsProcessorImpl implements LabelsProcessor {
  final _float32x4Zeroes = Float32x4.zero();
  final _float32x4Ones = Float32x4.splat(1.0);

  @override
  MLVector makeLabelsOneVsAll(MLVector origLabels, double targetLabel) {
    if (MLVector is Float32x4Vector) {
      return _makeFloat32x4LabelsOneVsAll(origLabels, targetLabel);
    }
    throw UnimplementedError();
  }

  MLVector _makeFloat32x4LabelsOneVsAll(MLVector origLabels, double targetLabel) {
//    final targetAsFloat32x4 = Float32x4.splat(targetLabel);
//    return origLabels
//        .fastMap((Float32x4 element, int start, int end) => element.equal(targetAsFloat32x4)
//        .select(_float32x4Ones, _float32x4Zeroes));
    throw UnimplementedError();
  }
}
