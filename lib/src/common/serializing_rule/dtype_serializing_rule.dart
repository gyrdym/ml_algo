import 'package:ml_linalg/dtype.dart';
import 'package:quiver/collection.dart';

BiMap<DType, String> get dtypeSerializingRule => BiMap()..addAll({
  DType.float32: 'float32',
  DType.float64: 'float64',
});
