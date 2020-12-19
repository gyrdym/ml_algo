import 'package:json_annotation/json_annotation.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/distance_type_to_json.dart';
import 'package:ml_linalg/from_distance_type_json.dart';

class DistanceTypeJsonConverter implements JsonConverter<Distance, String> {
  const DistanceTypeJsonConverter();

  @override
  Distance fromJson(String json) => fromDistanceTypeJson(json);

  @override
  String toJson(Distance distance) => distanceTypeToJson(distance);
}
