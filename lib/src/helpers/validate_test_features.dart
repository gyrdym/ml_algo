import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';

void validateTestFeatures(DataFrame features, DType dtype) {
  if (!features.toMatrix(dtype).hasData) {
    throw Exception('No features provided');
  }
}