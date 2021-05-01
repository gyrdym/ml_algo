import 'package:json_annotation/json_annotation.dart';
import 'package:ml_linalg/from_matrix_json.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:ml_linalg/matrix.dart';

class MatrixJsonConverterNullable implements JsonConverter<Matrix?, Map<String, dynamic>?> {
  const MatrixJsonConverterNullable();

  @override
  Matrix? fromJson(Map<String, dynamic>? json) => json?.isNotEmpty == true
      ? fromMatrixJson(json)
      : null;

  @override
  Map<String, dynamic>? toJson(Matrix? matrix) => matrixToJson(matrix) ?? {};
}
