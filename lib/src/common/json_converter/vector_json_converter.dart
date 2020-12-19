import 'package:json_annotation/json_annotation.dart';
import 'package:ml_linalg/from_vector_json.dart';
import 'package:ml_linalg/vector.dart';
import 'package:ml_linalg/vector_to_json.dart';

class VectorJsonConverter implements JsonConverter<Vector, Map<String, dynamic>> {
  const VectorJsonConverter();

  @override
  Vector fromJson(Map<String, dynamic> json) => fromVectorJson(json);

  @override
  Map<String, dynamic> toJson(Vector vector) => vectorToJson(vector);
}
