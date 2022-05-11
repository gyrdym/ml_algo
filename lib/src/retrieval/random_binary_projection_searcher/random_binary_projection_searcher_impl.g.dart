// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'random_binary_projection_searcher_impl.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

RandomBinaryProjectionSearcherImpl _$RandomBinaryProjectionSearcherImplFromJson(
    Map<String, dynamic> json) {
  return $checkedNew('RandomBinaryProjectionSearcherImpl', json, () {
    $checkKeys(json, allowedKeys: const ['D', 'H', 'P', 'R', 'B']);
    final val = RandomBinaryProjectionSearcherImpl(
      $checkedConvert(
          json, 'H', (v) => (v as List<dynamic>).map((e) => e as String)),
      $checkedConvert(
          json, 'P', (v) => Matrix.fromJson(v as Map<String, dynamic>)),
      $checkedConvert(json, 'D', (v) => v as int),
    );
    $checkedConvert(json, 'R',
        (v) => val.randomVectors = Matrix.fromJson(v as Map<String, dynamic>));
    $checkedConvert(
        json,
        'B',
        (v) => val.bins = (v as Map<String, dynamic>).map(
              (k, e) => MapEntry(int.parse(k),
                  (e as List<dynamic>).map((e) => e as int).toList()),
            ));
    return val;
  }, fieldKeyMap: const {
    'header': 'H',
    'points': 'P',
    'digitCapacity': 'D',
    'randomVectors': 'R',
    'bins': 'B'
  });
}

Map<String, dynamic> _$RandomBinaryProjectionSearcherImplToJson(
        RandomBinaryProjectionSearcherImpl instance) =>
    <String, dynamic>{
      'D': instance.digitCapacity,
      'H': instance.header.toList(),
      'P': instance.points.toJson(),
      'R': instance.randomVectors.toJson(),
      'B': instance.bins.map((k, e) => MapEntry(k.toString(), e)),
    };
