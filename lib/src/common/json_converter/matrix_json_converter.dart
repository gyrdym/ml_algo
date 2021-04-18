import 'package:json_annotation/json_annotation.dart';
import 'package:ml_linalg/from_matrix_json.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:ml_linalg/matrix.dart';

class MatrixJsonConverter implements JsonConverter<Matrix, Map<String, dynamic>> {
  const MatrixJsonConverter();

  @override
  Matrix fromJson(Map<String, dynamic> json) => fromMatrixJson(json)!;

  @override
  Map<String, dynamic> toJson(Matrix matrix) => matrixToJson(matrix)!;
}
